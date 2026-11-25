//! `rpython/rlib/_rweakkeydict.py` `WeakKeyDictRepr`.
//!
//! Values can outlive their keys until the dict is manipulated: a dead
//! weakref still holds a strong `WEAKDICTENTRY.value` until `ll_valid`
//! runs on that slot and stores `NULLVALUE`. Avoid this table when the
//! values keep large amounts of memory.
//!
//! `WEAKDICT` is `num_items`, `resize_counter`, `entries`. Each
//! `WEAKDICTENTRY` is `(key: WeakRefPtr, value: OBJECTPTR, f_hash: Signed)` —
//! no `f_valid`. `ll_valid` is a live `weakref_deref` of `key`, and a dead
//! weakref clears `value`. `ll_everused` is a non-null key word, so
//! `ll_set_null` stores `dead_wref` to keep the slot occupied.
//! `ll_hash` is `rdict.py` `ll_hash_from_cache` (`entries[i].f_hash`).
//! `ll_keyhash` is `compute_identity_hash`. `keyeq` is `ll_keyeq`
//! (deref of the stored weakref equals the probe object). `paranoia` is
//! false. `no_direct_compare` is true: the stored key is a `WeakRefPtr` and
//! the probe is the live object, so `ll_dict_lookup` never pointer-compares
//! them. Lookup and resize are `rdict.py` `ll_dict_lookup` / `ll_dict_resize`
//! (`DICT_INITSIZE` is 8).
//!
//! Entries are `_ll_malloc_entries`: collecting nursery malloc
//! (`try_gc_alloc_collecting_rooted`), matching `lltype.malloc(ENTRIES, n,
//! zero=True)` of a `GcArray`. `ll_new_weakdict` allocates the `WEAKDICT`
//! header the same collecting way (`lltype.malloc(WEAKDICT)`). A prebuilt
//! table (`WeakKeyDictRepr.convert_const`) uses `ll_new_weakdict_prebuilt`
//! (`alloc_raw`: `try_gc_alloc_stable_raw`, `GCFLAG_NO_HEAP_PTRS`, write
//! barrier into `prebuilt_root_objects`). `_ll_free_entries` is a no-op.

use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use crate::lltype::{GcType, TypeIdCell};
use crate::pyobject::PyObjectRef;
use crate::rdict::{self, RDict};
use crate::weakref::Weakref;

use rdict::{DICT_INITSIZE, HIGHEST_BIT};

/// `gctypelayout.py` `TypeLayoutBuilder.get_type_id` identity for `weakkeydict`.
pub static WEAKKEYDICT_GC_TYPE_ID_CELL: TypeIdCell = TypeIdCell::auto();
/// `gctypelayout.py` `TypeLayoutBuilder.get_type_id` identity for the key-dict
/// `WEAKDICTENTRYARRAY`.
pub static WEAKKEYDICT_ENTRIES_GC_TYPE_ID_CELL: TypeIdCell = TypeIdCell::auto();

static WEAKKEYDICT_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);
static WEAKKEYDICT_ENTRIES_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);

pub fn set_weakkeydict_gc_type_id(id: u32) {
    WEAKKEYDICT_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

pub fn set_weakkeydict_entries_gc_type_id(id: u32) {
    WEAKKEYDICT_ENTRIES_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

pub fn weakkeydict_gc_type_id() -> u32 {
    WEAKKEYDICT_GC_TYPE_ID.load(Ordering::Relaxed)
}

fn weakkeydict_entries_gc_type_id() -> u32 {
    WEAKKEYDICT_ENTRIES_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `WEAKDICTENTRY`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeakKeyDictEntry {
    pub key: *mut Weakref,
    pub value: PyObjectRef,
    pub f_hash: isize,
}

/// `WEAKDICTENTRYARRAY`. `length` is `len(entries)`.
#[repr(C)]
pub struct WeakKeyDictEntries {
    pub length: usize,
    pub items: [WeakKeyDictEntry; 0],
}

/// MiniMark-allocated by `alloc_entries` with the tid from
/// `weakkeydict_entries_gc_type_id`; this impl declares the gckind only.
impl GcType for WeakKeyDictEntries {
    fn type_id() -> u32 {
        u32::MAX
    }
    const SIZE: usize = 0;
}

/// `WEAKDICT` / `weakkeydict`.
#[repr(C)]
pub struct WeakKeyDict {
    pub num_items: isize,
    pub resize_counter: isize,
    pub entries: *mut WeakKeyDictEntries,
}

impl GcType for WeakKeyDict {
    fn type_id() -> u32 {
        match weakkeydict_gc_type_id() {
            0 => 0,
            id => id,
        }
    }
    const SIZE: usize = std::mem::size_of::<Self>();
}

/// Prebuilt `WEAKDICT` (`WeakKeyDictRepr.convert_const`): old-gen, non-moving
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
    if crate::gc_hook::gc_alloc_failed_pending() {
        return std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<usize>())
        .expect("weakkeydict layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    raw as crate::gc_hook::GCREF
}

fn entries_bytes(n: usize) -> usize {
    std::mem::offset_of!(WeakKeyDictEntries, items)
        .checked_add(
            n.checked_mul(std::mem::size_of::<WeakKeyDictEntry>())
                .expect("weakkeydict entries size"),
        )
        .expect("weakkeydict entries size")
}

/// Collecting nursery malloc (`try_gc_alloc_collecting_rooted`) with the
/// no-GC host fallback `alloc_entries` uses.
fn alloc_collecting_or_host(tid: u32, bytes: usize) -> crate::gc_hook::GCREF {
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
            .allocated_or_raise()
            .unwrap_or(std::ptr::null_mut());
        if crate::gc_hook::gc_alloc_failed_pending() {
            return std::ptr::null_mut();
        }
        if !raw.is_null() {
            unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
            if needs_write_barrier {
                crate::gc_hook::try_gc_write_barrier_managed(raw);
            }
            return raw;
        }
    }
    let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<usize>())
        .expect("weakkeydict layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    raw as crate::gc_hook::GCREF
}

/// `_ll_malloc_entries`: `malloc(ENTRIES, n, zero=True)` of a `GcArray`.
fn alloc_entries(n: usize) -> *mut WeakKeyDictEntries {
    let bytes = entries_bytes(n);
    let tid = weakkeydict_entries_gc_type_id();
    let raw = alloc_collecting_or_host(tid, bytes);
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    let entries = raw as *mut WeakKeyDictEntries;
    unsafe { (*entries).length = n };
    entries
}

fn items(entries: *mut WeakKeyDictEntries) -> *mut WeakKeyDictEntry {
    unsafe { std::ptr::addr_of_mut!((*entries).items).cast() }
}

fn entry(entries: *mut WeakKeyDictEntries, i: usize) -> *mut WeakKeyDictEntry {
    unsafe { items(entries).add(i) }
}

/// `ll_everused`: `bool(entries[i].key)`.
fn everused(entries: *mut WeakKeyDictEntries, i: usize) -> bool {
    unsafe { !(*entry(entries, i)).key.is_null() }
}

/// `ll_valid`. A dead weakref still holding a strong value is cleared to
/// `NULLVALUE`.
fn valid(entries: *mut WeakKeyDictEntries, i: usize) -> bool {
    let key = unsafe { (*entry(entries, i)).key };
    if key.is_null() {
        return false;
    }
    if !unsafe { crate::weakref::w_weakref_deref(key) }.is_null() {
        return true;
    }
    barrier_entries(entries, i);
    unsafe { (*entry(entries, i)).value = std::ptr::null_mut() };
    false
}

/// `WEAKDICTENTRYARRAY` adt `hash`: `ll_hash_from_cache`.
fn ll_hash(entries: *mut WeakKeyDictEntries, i: usize) -> isize {
    unsafe { (*entry(entries, i)).f_hash }
}

/// `compute_identity_hash(llkey)`.
fn ll_keyhash(llkey: PyObjectRef) -> isize {
    crate::gc_hook::gc_identity_hash(llkey as usize) as isize
}

/// `ll_keyeq(d, weakkey1, realkey2)`.
fn ll_keyeq(weakkey1: *mut Weakref, realkey2: PyObjectRef) -> bool {
    if weakkey1.is_null() {
        debug_assert!(!realkey2.is_null());
        return false;
    }
    unsafe { crate::weakref::w_weakref_deref(weakkey1) == realkey2 }
}

/// `llmemory.dead_wref`. A non-null `WeakRefPtr` whose deref is null, so
/// `everused()` stays true after `ll_set_null`.
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

/// `setarrayitem_gc` write barrier on `WEAKDICTENTRYARRAY`.
fn barrier_entries(entries: *mut WeakKeyDictEntries, i: usize) {
    if !entries.is_null() {
        crate::gc_hook::try_gc_write_barrier_from_array(entries as crate::gc_hook::GCREF, i);
    }
}

/// `setfield_gc` write barrier on `WEAKDICT.entries`.
fn barrier_dict(d: *mut WeakKeyDict) {
    if !d.is_null() {
        crate::gc_hook::try_gc_write_barrier(d as crate::gc_hook::GCREF);
    }
}

/// `setfield_gc` write barrier on `WEAKDICT`.
pub fn barrier_weakkeydict(d: *mut WeakKeyDict) {
    barrier_dict(d);
}

/// `rordereddict.rs` `RDict::pin_table_and_entries`.
///
/// Pins `d` when it is a GC object (`try_gc_owns_object`), then pins
/// `entries` when non-null. Returns `(dict_slot, entries_slot)` recorded
/// at each pin: `dict_slot` is `None` when `d` is not a GC object
/// (host-fallback test dicts), and the caller keeps the raw pointer.
/// `entries_slot` is the slot `entries` was published into, or the current
/// stack length when `entries` is null. A nursery `WEAKDICT` moves, so
/// callers read both back after a collecting malloc (`pinned_dict` /
/// `pinned_entries`; `rordereddict.rs` `pinned_entries`).
fn pin_table_and_entries(
    d: *mut WeakKeyDict,
    entries: *mut WeakKeyDictEntries,
) -> (Option<usize>, usize) {
    let dict_slot = if !d.is_null() {
        barrier_dict(d);
        if crate::gc_hook::try_gc_owns_object(d as crate::gc_hook::GCREF) {
            let _ = crate::gc_roots::pin_root(d as PyObjectRef);
            Some(crate::gc_roots::shadow_stack_len() - 1)
        } else {
            None
        }
    } else {
        None
    };
    let entries_slot = crate::gc_roots::shadow_stack_len();
    if !entries.is_null() {
        crate::gc_hook::try_gc_write_barrier(entries as crate::gc_hook::GCREF);
        let _ = crate::gc_roots::pin_root(entries as PyObjectRef);
    }
    (dict_slot, entries_slot)
}

/// `WEAKDICT` read back from the slot [`pin_table_and_entries`] published,
/// or `raw` when the dict is not a GC object.
fn pinned_dict(dict_slot: Option<usize>, raw: *mut WeakKeyDict) -> *mut WeakKeyDict {
    match dict_slot {
        Some(slot) => crate::gc_roots::shadow_stack_get(slot) as *mut WeakKeyDict,
        None => raw,
    }
}

/// `d.entries` read back from the slot [`pin_table_and_entries`] answered
/// (`rordereddict.rs` `pinned_entries`).
fn pinned_entries(slot: usize) -> *mut WeakKeyDictEntries {
    crate::gc_roots::shadow_stack_get(slot) as *mut WeakKeyDictEntries
}

/// `ll_setitem_fast` on `d.entries[i]`.
fn store_entry(
    entries: *mut WeakKeyDictEntries,
    i: usize,
    key: *mut Weakref,
    value: PyObjectRef,
    f_hash: isize,
) {
    barrier_entries(entries, i);
    unsafe {
        (*entry(entries, i)).key = key;
        (*entry(entries, i)).value = value;
        (*entry(entries, i)).f_hash = f_hash;
    }
}

impl RDict for WeakKeyDict {
    type Key = *mut Weakref;
    type LookupKey = PyObjectRef;
    type Value = PyObjectRef;
    type Entries = WeakKeyDictEntries;
    type ResizePin = (Option<usize>, usize);

    fn num_items(d: *mut Self) -> isize {
        unsafe { (*d).num_items }
    }

    fn set_num_items(d: *mut Self, value: isize) {
        unsafe { (*d).num_items = value };
    }

    fn resize_counter(d: *mut Self) -> isize {
        unsafe { (*d).resize_counter }
    }

    fn set_resize_counter(d: *mut Self, value: isize) {
        unsafe { (*d).resize_counter = value };
    }

    fn entries(d: *mut Self) -> *mut Self::Entries {
        unsafe { (*d).entries }
    }

    fn set_entries(d: *mut Self, entries: *mut Self::Entries) {
        barrier_dict(d);
        unsafe { (*d).entries = entries };
    }

    fn len(entries: *mut Self::Entries) -> usize {
        unsafe { (*entries).length }
    }

    fn valid(entries: *mut Self::Entries, i: usize) -> bool {
        valid(entries, i)
    }

    fn everused(entries: *mut Self::Entries, i: usize) -> bool {
        everused(entries, i)
    }

    fn hash(entries: *mut Self::Entries, i: usize) -> isize {
        ll_hash(entries, i)
    }

    fn key(entries: *mut Self::Entries, i: usize) -> Self::Key {
        unsafe { (*entry(entries, i)).key }
    }

    fn value(entries: *mut Self::Entries, i: usize) -> Self::Value {
        unsafe { (*entry(entries, i)).value }
    }

    fn store(
        entries: *mut Self::Entries,
        i: usize,
        key: Self::Key,
        value: Self::Value,
        hash: isize,
    ) {
        store_entry(entries, i, key, value, hash);
    }

    fn mark_deleted(entries: *mut Self::Entries, i: usize) {
        let hash = ll_hash(entries, i);
        store_entry(entries, i, dead_wref(), std::ptr::null_mut(), hash);
    }

    fn direct_compare(_key1: Self::Key, _key2: Self::LookupKey) -> bool {
        false
    }

    fn keyeq(&self, key1: Self::Key, key2: Self::LookupKey) -> bool {
        ll_keyeq(key1, key2)
    }

    fn allocate_entries(n: usize) -> *mut Self::Entries {
        alloc_entries(n)
    }

    fn pin_for_resize(d: *mut Self, entries: *mut Self::Entries) -> Self::ResizePin {
        pin_table_and_entries(d, entries)
    }

    fn reload_after_allocate(
        d: *mut Self,
        pin: &Self::ResizePin,
    ) -> (*mut Self, *mut Self::Entries) {
        (pinned_dict(pin.0, d), pinned_entries(pin.1))
    }
}

/// `ll_new_weakdict`. Ordinary collectable malloc (`lltype.malloc(WEAKDICT)`).
#[majit_macros::dont_look_inside]
pub fn ll_new_weakdict() -> *mut WeakKeyDict {
    let bytes = std::mem::size_of::<WeakKeyDict>();
    let tid = weakkeydict_gc_type_id();
    let raw = alloc_collecting_or_host(tid, bytes);
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    let _roots = crate::gc_roots::push_roots();
    let (dict_slot, _) = pin_table_and_entries(raw as *mut WeakKeyDict, std::ptr::null_mut());
    let entries = alloc_entries(DICT_INITSIZE);
    if crate::gc_hook::gc_alloc_failed_pending() && entries.is_null() {
        return std::ptr::null_mut();
    }
    let d = pinned_dict(dict_slot, raw as *mut WeakKeyDict);
    barrier_dict(d);
    unsafe {
        (*d).entries = entries;
        (*d).num_items = 0;
        (*d).resize_counter = (DICT_INITSIZE * 2) as isize;
    }
    d
}

/// `WeakKeyDictRepr.convert_const`: a prebuilt `WEAKDICT`.
/// `alloc_raw` (`try_gc_alloc_stable_raw` + `GCFLAG_NO_HEAP_PTRS` + write
/// barrier) enrolls it in `prebuilt_root_objects`.
#[majit_macros::dont_look_inside]
pub fn ll_new_weakdict_prebuilt() -> *mut WeakKeyDict {
    let raw = alloc_raw(weakkeydict_gc_type_id(), std::mem::size_of::<WeakKeyDict>());
    if crate::gc_hook::gc_alloc_failed_pending() && raw.is_null() {
        return std::ptr::null_mut();
    }
    let _roots = crate::gc_roots::push_roots();
    let (dict_slot, _) = pin_table_and_entries(raw as *mut WeakKeyDict, std::ptr::null_mut());
    let entries = alloc_entries(DICT_INITSIZE);
    if crate::gc_hook::gc_alloc_failed_pending() && entries.is_null() {
        return std::ptr::null_mut();
    }
    let d = pinned_dict(dict_slot, raw as *mut WeakKeyDict);
    barrier_dict(d);
    unsafe {
        (*d).entries = entries;
        (*d).num_items = 0;
        (*d).resize_counter = (DICT_INITSIZE * 2) as isize;
    }
    d
}

fn ll_dict_lookup_iff(d: &WeakKeyDict, key: PyObjectRef, _hash: isize) -> bool {
    majit_rlib::jit::isvirtual(d) && majit_rlib::jit::isconstant(&key)
}

/// `rpython.rtyper.lltypesystem.rdict.ll_dict_lookup`.
#[majit_macros::oopspec("dict.lookup")]
#[majit_macros::look_inside_iff(ll_dict_lookup_iff)]
fn ll_dict_lookup(d: &WeakKeyDict, key: PyObjectRef, hash: isize) -> usize {
    rdict::ll_dict_lookup(d, key, hash)
}

impl WeakKeyDict {
    /// `ll_get(d, llkey)`. Null is the missing value.
    #[majit_macros::dont_look_inside]
    pub fn ll_get(&self, llkey: PyObjectRef) -> PyObjectRef {
        let hash = ll_keyhash(llkey);
        let i = ll_dict_lookup(self, llkey, hash) & (HIGHEST_BIT - 1);
        unsafe { (*entry(self.entries, i)).value }
    }

    /// `ll_set`. A null `llvalue` is `None` and dispatches to `ll_set_null`.
    #[majit_macros::dont_look_inside]
    pub fn ll_set(&mut self, llkey: PyObjectRef, llvalue: PyObjectRef) -> bool {
        if !llvalue.is_null() {
            self.ll_set_nonnull(llkey, llvalue)
        } else {
            self.ll_set_null(llkey);
            true
        }
    }

    /// `ll_set_null`. Stores `dead_wref` so `everused()` stays true.
    #[majit_macros::dont_look_inside]
    pub fn ll_set_null(&mut self, llkey: PyObjectRef) {
        let hash = ll_keyhash(llkey);
        let i = ll_dict_lookup(self, llkey, hash) & (HIGHEST_BIT - 1);
        if everused(self.entries, i) {
            <Self as RDict>::mark_deleted(self.entries, i);
        }
    }

    /// `ll_set_nonnull`. `hash = compute_identity_hash(llkey)` then
    /// `weakref_create(llkey)` (GC effects), then the store with that hash.
    /// `d`, `llkey`, and `llvalue` are livevars across that malloc
    /// (`framework.py` `push_roots` / `pop_roots`). Reload `d` from its
    /// shadow-stack slot after `w_weakref_new` (`ll_new_weakdict`;
    /// `rordereddict.rs` `RDict::pin_table_and_entries` / `pinned_entries`).
    #[majit_macros::dont_look_inside]
    pub fn ll_set_nonnull(&mut self, llkey: PyObjectRef, llvalue: PyObjectRef) -> bool {
        let _roots = crate::gc_roots::push_roots();
        let d_raw = self as *mut WeakKeyDict;
        barrier_weakkeydict(d_raw);
        let (dict_slot, pair) =
            if crate::gc_hook::try_gc_owns_object(d_raw as crate::gc_hook::GCREF) {
                let base = crate::gc_roots::pin_roots(&[d_raw as PyObjectRef, llvalue, llkey]);
                (Some(base), base + 1)
            } else {
                (None, crate::gc_roots::pin_roots(&[llvalue, llkey]))
            };
        let hash = ll_keyhash(crate::gc_roots::shadow_stack_get(pair + 1));
        let keyref =
            unsafe { crate::weakref::w_weakref_new(crate::gc_roots::shadow_stack_get(pair + 1)) };
        if crate::gc_hook::gc_alloc_failed_pending() && keyref.is_null() {
            return false;
        }
        let llvalue = crate::gc_roots::shadow_stack_get(pair);
        let llkey = crate::gc_roots::shadow_stack_get(pair + 1);
        let d = pinned_dict(dict_slot, d_raw);
        unsafe { (*d).ll_set_nonnull_keyref(llkey, llvalue, keyref, hash) }
    }

    /// Lookup/insert half of `ll_set_nonnull` after `weakref_create`.
    #[majit_macros::dont_look_inside]
    fn ll_set_nonnull_keyref(
        &mut self,
        llkey: PyObjectRef,
        llvalue: PyObjectRef,
        keyref: *mut Weakref,
        hash: isize,
    ) -> bool {
        debug_assert!(!keyref.is_null());
        let found = ll_dict_lookup(self, llkey, hash);
        let i = found & (HIGHEST_BIT - 1);
        let was_everused = everused(self.entries, i);
        store_entry(self.entries, i, keyref, llvalue, hash);
        if !was_everused {
            self.resize_counter -= 3;
            if self.resize_counter <= 0 {
                ll_weakdict_resize(self as *mut WeakKeyDict);
                return !crate::gc_hook::gc_alloc_failed_pending();
            }
        }
        true
    }

    /// `ll_length`. Slow; `ll_update_num_items` then `d.num_items`.
    #[majit_macros::dont_look_inside]
    pub fn ll_length(&mut self) -> isize {
        self.ll_update_num_items();
        self.num_items
    }

    pub fn count_valid(&self) -> usize {
        if self.entries.is_null() {
            return 0;
        }
        let n = unsafe { (*self.entries).length };
        (0..n).filter(|&i| valid(self.entries, i)).count()
    }

    fn ll_update_num_items(&mut self) {
        let entries = self.entries;
        let n = unsafe { (*entries).length };
        let mut num_items = 0isize;
        for i in 0..n {
            if valid(entries, i) {
                num_items += 1;
            }
        }
        self.num_items = num_items;
    }
}

/// `ll_weakdict_resize` then `ll_dict_resize`.
fn ll_weakdict_resize(d: *mut WeakKeyDict) {
    unsafe { (*d).ll_update_num_items() };
    rdict::ll_dict_resize(d);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdict::ll_dict_lookup;

    fn dict() -> &'static mut WeakKeyDict {
        unsafe { &mut *ll_new_weakdict() }
    }

    #[test]
    fn miss_returns_null_and_a_second_key_does_not_clobber_it() {
        let d = dict();
        let key = 0x1111 as PyObjectRef;
        let obj = 0x2222 as PyObjectRef;
        d.ll_set_nonnull(key, obj);
        assert_eq!(d.ll_get(key), obj);
        assert!(d.ll_get(0x1112 as PyObjectRef).is_null());
        assert_eq!(d.count_valid(), 1);
        assert_eq!(d.ll_length(), 1);
    }

    /// Six new slots drive `resize_counter` from `DICT_INITSIZE * 2` through
    /// zero (`ll_set_nonnull` subtracts 3). Every key must still be found.
    #[test]
    fn resize_keeps_live_weak_keys() {
        let d = dict();
        let mut objs = Vec::new();
        for n in 0..6 {
            let key = (0x3000 + n) as PyObjectRef;
            let obj = (0x4000 + n) as PyObjectRef;
            d.ll_set_nonnull(key, obj);
            objs.push((key, obj));
        }
        assert!(unsafe { (*d.entries).length } > DICT_INITSIZE);
        for (key, obj) in objs {
            assert_eq!(d.ll_get(key), obj);
        }
        assert_eq!(d.ll_length(), 6);
    }

    /// `ll_set` with a live value is `ll_set_nonnull`.
    #[test]
    fn ll_set_stores_a_live_value() {
        let d = dict();
        let key = 0x4444 as PyObjectRef;
        let obj = 0x5555 as PyObjectRef;
        d.ll_set(key, obj);
        assert_eq!(d.ll_get(key), obj);
        assert_eq!(d.count_valid(), 1);
    }

    /// `ll_set_null` stores `dead_wref` so `everused()` stays true.
    /// `ll_set` with a null value dispatches here.
    #[test]
    fn ll_set_null_keeps_everused() {
        let d = dict();
        let key = 0x5555 as PyObjectRef;
        let obj = 0x6666 as PyObjectRef;
        d.ll_set(key, obj);
        assert_eq!(d.ll_get(key), obj);

        d.ll_set(key, std::ptr::null_mut());
        assert!(d.ll_get(key).is_null());
        assert_eq!(d.count_valid(), 0);
        assert_eq!(d.ll_length(), 0);

        let i = ll_dict_lookup(d, key, ll_keyhash(key)) & (HIGHEST_BIT - 1);
        assert!(everused(d.entries, i));
        assert!(!valid(d.entries, i));
        let slot = unsafe { *entry(d.entries, i) };
        assert_eq!(slot.key, dead_wref());
        assert!(slot.value.is_null());
        assert!(unsafe { crate::weakref::w_weakref_deref(slot.key) }.is_null());

        let other = 0x5557 as PyObjectRef;
        let other_obj = 0x6667 as PyObjectRef;
        d.ll_set(other, other_obj);
        assert_eq!(d.ll_get(other), other_obj);
        assert!(d.ll_get(key).is_null());
    }

    /// `ll_set_null` on a key that was never stored does not occupy a slot.
    #[test]
    fn ll_set_null_on_unused_key_does_not_occupy_a_slot() {
        let d = dict();
        let key = 0x7777 as PyObjectRef;
        d.ll_set_null(key);
        assert!(d.ll_get(key).is_null());
        assert_eq!(d.count_valid(), 0);
        let i = ll_dict_lookup(d, key, ll_keyhash(key)) & (HIGHEST_BIT - 1);
        assert!(!everused(d.entries, i));
    }

    /// Identity keys: two distinct pointers are two slots even when the
    /// stored values compare equal.
    #[test]
    fn identity_keys_do_not_collapse() {
        let d = dict();
        let k1 = 0x8888 as PyObjectRef;
        let k2 = 0x8889 as PyObjectRef;
        let v1 = 0x9999 as PyObjectRef;
        let v2 = 0x999A as PyObjectRef;
        d.ll_set(k1, v1);
        d.ll_set(k2, v2);
        assert_eq!(d.ll_get(k1), v1);
        assert_eq!(d.ll_get(k2), v2);
        assert_eq!(d.count_valid(), 2);
    }

    /// `ll_valid` on a dead weakref clears `value` to `NULLVALUE`.
    #[test]
    fn ll_valid_clears_value_of_a_dead_key() {
        let d = dict();
        let key = 0xAAAA as PyObjectRef;
        let obj = 0xBBBB as PyObjectRef;
        d.ll_set_nonnull(key, obj);
        let i = ll_dict_lookup(d, key, ll_keyhash(key)) & (HIGHEST_BIT - 1);
        let wref = unsafe { (*entry(d.entries, i)).key };
        unsafe { (*wref).weakptr = std::ptr::null_mut() };
        assert!(d.ll_get(key).is_null());
        assert!(unsafe { (*entry(d.entries, i)).value }.is_null());
        assert!(everused(d.entries, i));
        assert!(!valid(d.entries, i));
    }
}
