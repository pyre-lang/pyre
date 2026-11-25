//! Shared low-level dictionary operations from
//! `rpython/rtyper/lltypesystem/rdict.py`.

pub(crate) const DICT_INITSIZE: usize = 8;
pub(crate) const HIGHEST_BIT: usize = 1usize << (usize::BITS - 1);
const PERTURB_SHIFT: u32 = 5;

/// Entry and dictionary ADT methods consumed by the shared `rdict` routines.
pub trait RDict: Sized {
    type Key: Copy;
    type LookupKey: Copy;
    type Value: Copy;
    type Entries;
    type ResizePin;

    fn num_items(d: *mut Self) -> isize;
    fn set_num_items(d: *mut Self, value: isize);
    fn resize_counter(d: *mut Self) -> isize;
    fn set_resize_counter(d: *mut Self, value: isize);
    fn entries(d: *mut Self) -> *mut Self::Entries;
    fn set_entries(d: *mut Self, entries: *mut Self::Entries);

    fn len(entries: *mut Self::Entries) -> usize;
    fn valid(entries: *mut Self::Entries, i: usize) -> bool;
    fn everused(entries: *mut Self::Entries, i: usize) -> bool;
    fn hash(entries: *mut Self::Entries, i: usize) -> isize;
    fn key(entries: *mut Self::Entries, i: usize) -> Self::Key;
    fn value(entries: *mut Self::Entries, i: usize) -> Self::Value;
    fn store(
        entries: *mut Self::Entries,
        i: usize,
        key: Self::Key,
        value: Self::Value,
        hash: isize,
    );
    fn mark_deleted(entries: *mut Self::Entries, i: usize);

    fn direct_compare(key1: Self::Key, key2: Self::LookupKey) -> bool;
    fn keyeq(&self, key1: Self::Key, key2: Self::LookupKey) -> bool;

    fn allocate_entries(n: usize) -> *mut Self::Entries;
    fn pin_for_resize(d: *mut Self, entries: *mut Self::Entries) -> Self::ResizePin;
    fn reload_after_allocate(
        d: *mut Self,
        pin: &Self::ResizePin,
    ) -> (*mut Self, *mut Self::Entries);
}

pub(crate) trait RDictProbe<D: RDict> {
    fn direct_compare(&self, stored: D::Key) -> bool;
    fn keyeq(&self, d: &D, stored: D::Key) -> bool;
}

struct KeyProbe<K>(K);

impl<D: RDict> RDictProbe<D> for KeyProbe<D::LookupKey> {
    fn direct_compare(&self, stored: D::Key) -> bool {
        D::direct_compare(stored, self.0)
    }

    fn keyeq(&self, d: &D, stored: D::Key) -> bool {
        d.keyeq(stored, self.0)
    }
}

/// `rpython.rtyper.lltypesystem.rdict.ll_dict_lookup`.
pub(crate) fn ll_dict_lookup<D: RDict>(d: &D, key: D::LookupKey, hash: isize) -> usize {
    ll_dict_lookup_with(d, &KeyProbe(key), hash)
}

/// `rpython.rtyper.lltypesystem.rdict.ll_dict_lookup`.
pub(crate) fn ll_dict_lookup_with<D, P>(d: &D, key: &P, hash: isize) -> usize
where
    D: RDict,
    P: RDictProbe<D>,
{
    let entries = D::entries(d as *const D as *mut D);
    let mask = D::len(entries) - 1;
    let mut i = (hash as usize) & mask;
    let mut freeslot: isize;

    if D::valid(entries, i) {
        let checkingkey = D::key(entries, i);
        if key.direct_compare(checkingkey) {
            return i;
        }
        if D::hash(entries, i) == hash && key.keyeq(d, checkingkey) {
            return i;
        }
        freeslot = -1;
    } else if D::everused(entries, i) {
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
        if !D::everused(entries, i) {
            if freeslot == -1 {
                freeslot = i as isize;
            }
            return (freeslot as usize) | HIGHEST_BIT;
        } else if D::valid(entries, i) {
            let checkingkey = D::key(entries, i);
            if key.direct_compare(checkingkey) {
                return i;
            }
            if D::hash(entries, i) == hash && key.keyeq(d, checkingkey) {
                return i;
            }
        } else if freeslot == -1 {
            freeslot = i as isize;
        }
        perturb >>= PERTURB_SHIFT;
    }
}

/// `rpython.rtyper.lltypesystem.rdict.ll_dict_lookup_clean`.
pub(crate) fn ll_dict_lookup_clean<D: RDict>(d: &D, hash: isize) -> usize {
    let entries = D::entries(d as *const D as *mut D);
    let mask = D::len(entries) - 1;
    let mut i = (hash as usize) & mask;
    let mut perturb = hash as usize;
    while D::everused(entries, i) {
        i = (i << 2)
            .wrapping_add(i)
            .wrapping_add(perturb)
            .wrapping_add(1)
            & mask;
        perturb >>= PERTURB_SHIFT;
    }
    i
}

/// `rpython.rtyper.lltypesystem.rdict.ll_dict_insertclean`.
pub(crate) fn ll_dict_insertclean<D: RDict>(d: *mut D, key: D::Key, value: D::Value, hash: isize) {
    let i = ll_dict_lookup_clean(unsafe { &*d }, hash);
    D::store(D::entries(d), i, key, value, hash);
    D::set_num_items(d, D::num_items(d) + 1);
    D::set_resize_counter(d, D::resize_counter(d) - 3);
}

/// `rpython.rtyper.lltypesystem.rdict.ll_dict_resize`.
#[majit_macros::oopspec("dict.resize")]
pub(crate) fn ll_dict_resize<D: RDict>(d: *mut D) {
    let num_extra = std::cmp::min(D::num_items(d) + 1, 30000);
    _ll_dict_resize_to(d, num_extra);
}

/// `rpython.rtyper.lltypesystem.rdict._ll_dict_resize_to`.
pub(crate) fn _ll_dict_resize_to<D: RDict>(d: *mut D, num_extra: isize) {
    let new_estimate = (D::num_items(d) + num_extra) * 2;
    let mut new_size = DICT_INITSIZE;
    while (new_size as isize) <= new_estimate {
        new_size *= 2;
    }
    let old_entries = D::entries(d);
    let old_size = D::len(old_entries);
    let _roots = crate::gc_roots::push_roots();
    let pin = D::pin_for_resize(d, old_entries);
    let new_entries = D::allocate_entries(new_size);
    if crate::gc_hook::gc_alloc_failed_pending() && new_entries.is_null() {
        return;
    }
    let (d, old_entries) = D::reload_after_allocate(d, &pin);
    D::set_entries(d, new_entries);
    D::set_num_items(d, 0);
    D::set_resize_counter(d, (new_size * 2) as isize);
    for i in 0..old_size {
        if D::valid(old_entries, i) {
            let hash = D::hash(old_entries, i);
            let key = D::key(old_entries, i);
            let value = D::value(old_entries, i);
            ll_dict_insertclean(d, key, value, hash);
        }
    }
}
