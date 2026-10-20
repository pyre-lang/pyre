//! `DICTENTRYARRAY = lltype.GcArray(DICTENTRY)` (`rordereddict.py get_ll_dict`).
//!
//! The array is allocated with `_ll_malloc_entries` (`malloc(ENTRIES, n, zero=True)`).
//! A zero item is an invalid entry: `f_valid` is false and the key and value are
//! the all-zero [`EntryDummy`].

use std::alloc::Layout;
use std::sync::atomic::{AtomicU32, Ordering};

/// One `d.entries` element — `odictentry` (`get_ll_dict` `entryfields`):
/// `key`, `f_valid`, `value`, `f_hash`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Entry<K, V> {
    pub key: K,
    pub f_valid: bool,
    pub value: V,
    pub f_hash: u64,
}

/// The value a deleted entry's key or value is reset to
/// (`must_clear_key` / `must_clear_value` store `nullptr`).
pub trait EntryDummy {
    fn dummy() -> Self;
}

impl EntryDummy for u64 {
    fn dummy() -> Self {
        0
    }
}

impl EntryDummy for i64 {
    fn dummy() -> Self {
        0
    }
}

impl EntryDummy for () {
    fn dummy() -> Self {}
}

/// `DICTENTRYARRAY`. `length` is `len(d.entries)`: the allocated item count.
/// The collector reads it as the varsize length and visits every item below it.
#[repr(C)]
pub struct GcEntries<K, V> {
    pub length: usize,
    /// Varsize tail. The allocation extends `length` items past this field.
    pub items: [Entry<K, V>; 0],
}

/// Byte offsets of the GC references inside one value of `Self`
/// (`gctypelayout` `varofstoptrs` contribution).
pub trait GcRefOffsets {
    const GC_REF_OFFSETS: &'static [usize];
}

impl GcRefOffsets for crate::pyobject::PyObjectRef {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for crate::dictmultiobject::ObjectKey {
    const GC_REF_OFFSETS: &'static [usize] =
        &[std::mem::offset_of!(crate::dictmultiobject::ObjectKey, obj)];
}

impl GcRefOffsets for crate::identitydict::IdentityKey {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for crate::dictmultiobject::BytesKey {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for crate::celldict::StrKey {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for i64 {
    const GC_REF_OFFSETS: &'static [usize] = &[];
}

impl GcRefOffsets for u64 {
    const GC_REF_OFFSETS: &'static [usize] = &[];
}

impl GcRefOffsets for () {
    const GC_REF_OFFSETS: &'static [usize] = &[];
}

/// Per-`(K, V)` GC type id of [`GcEntries`].
pub trait GcEntriesType {
    fn entries_gc_type_id() -> u32;
}

macro_rules! entries_gc_type_id {
    ($atomic:ident, $setter:ident, $getter:ident, $k:ty, $v:ty) => {
        static $atomic: AtomicU32 = AtomicU32::new(0);

        pub fn $setter(id: u32) {
            $atomic.store(id, Ordering::Relaxed);
        }

        #[majit_macros::dont_look_inside]
        pub fn $getter() -> u32 {
            $atomic.load(Ordering::Relaxed)
        }

        impl GcEntriesType for ($k, $v) {
            fn entries_gc_type_id() -> u32 {
                $getter()
            }
        }
    };
}

entries_gc_type_id!(
    OBJECT_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_object_key_pyobject_entries_gc_type_id,
    object_key_pyobject_entries_gc_type_id,
    crate::dictmultiobject::ObjectKey,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    I64_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_i64_pyobject_entries_gc_type_id,
    i64_pyobject_entries_gc_type_id,
    i64,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    BYTES_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_bytes_key_pyobject_entries_gc_type_id,
    bytes_key_pyobject_entries_gc_type_id,
    crate::dictmultiobject::BytesKey,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    OBJECT_KEY_UNIT_ENTRIES_GC_TYPE_ID,
    set_object_key_unit_entries_gc_type_id,
    object_key_unit_entries_gc_type_id,
    crate::dictmultiobject::ObjectKey,
    ()
);
entries_gc_type_id!(
    IDENTITY_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_identity_key_pyobject_entries_gc_type_id,
    identity_key_pyobject_entries_gc_type_id,
    crate::identitydict::IdentityKey,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    STR_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_str_key_pyobject_entries_gc_type_id,
    str_key_pyobject_entries_gc_type_id,
    crate::celldict::StrKey,
    crate::pyobject::PyObjectRef
);

/// GC-reference offsets inside one [`Entry`], relative to the item start.
pub fn entry_gc_ref_offsets<K: GcRefOffsets, V: GcRefOffsets>() -> Vec<usize> {
    let mut offsets = Vec::new();
    let key_base = std::mem::offset_of!(Entry<K, V>, key);
    for off in K::GC_REF_OFFSETS {
        offsets.push(key_base + off);
    }
    let value_base = std::mem::offset_of!(Entry<K, V>, value);
    for off in V::GC_REF_OFFSETS {
        offsets.push(value_base + off);
    }
    offsets
}

/// First item of a `DICTENTRYARRAY`. Null stays null.
pub fn entries_item_ptr<K, V>(entries: *mut GcEntries<K, V>) -> *mut Entry<K, V> {
    if entries.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { std::ptr::addr_of_mut!((*entries).items).cast() }
    }
}

/// `len(d.entries)`. A null array is length 0.
pub fn entries_allocated_len<K, V>(entries: *const GcEntries<K, V>) -> usize {
    if entries.is_null() {
        0
    } else {
        unsafe { (*entries).length }
    }
}

fn entries_payload_bytes<K, V>(n: usize) -> usize {
    std::mem::offset_of!(GcEntries<K, V>, items)
        .checked_add(
            n.checked_mul(std::mem::size_of::<Entry<K, V>>())
                .expect("entries allocation size"),
        )
        .expect("entries allocation size")
}

/// `_ll_malloc_entries`: `malloc(ENTRIES, n, zero=True)`.
///
/// `n == 0` is the empty array, represented by null (an empty `RDict` holds
/// no array until the first insert).
///
/// Upstream's `lltype.malloc` is young: a nursery `malloc_varsize` that
/// moves at the next minor collection. A caller holding the array address
/// or a `&K` / `&V` into it across a collection reads it back from a root.
/// An old-gen
/// `DICTENTRYARRAY` that holds a heap type's `__dict__` / `__weakref__`
/// GetSet copies (`w_objclass`) would enter the remembered set and promote
/// the type on every minor, so the block is never born old on purpose.
pub fn alloc_entries<K, V>(n: usize) -> *mut GcEntries<K, V>
where
    (K, V): GcEntriesType,
{
    if n == 0 {
        return std::ptr::null_mut();
    }
    let bytes = entries_payload_bytes::<K, V>(n);
    let tid = <(K, V)>::entries_gc_type_id();
    if tid != 0 {
        // `_ll_malloc_entries`: `malloc(ENTRIES, n, zero=True)` is
        // `incminimark.py malloc_varsize`: a nursery bump, and
        // `external_malloc(..., alloc_young=True)` past the large-object
        // threshold. A full nursery takes the same young external birth
        // instead of `collect_and_reserve`: dict insertion callers hold raw
        // GC pointers on the Rust stack across `ll_dict_grow`, so nothing
        // collects here and the threshold becomes the deferred major
        // request the interpreter safepoint serves.
        let young = majit_gc::try_alloc_nursery_bump_typed(tid, bytes).0 as crate::gc_hook::GCREF;
        let raw = if young.is_null() {
            crate::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(tid, bytes)
        } else {
            young
        };
        if !raw.is_null() {
            // Neither malloc zero-fills; `malloc(..., zero=True)` does.
            // `GCREF` points at a zero-sized `GCREFOpaque`, so the clear
            // goes through `*mut u8`; on the GCREF it writes no bytes.
            // The barrier is for a type whose young birth is refused and
            // is born old, then filled with young items before the next
            // minor.
            unsafe {
                std::ptr::write_bytes(raw as *mut u8, 0, bytes);
            }
            let entries = raw as *mut GcEntries<K, V>;
            unsafe {
                (*entries).length = n;
            }
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            return entries;
        }
    }
    // `tid == 0` or no hook (unit tests, `pyre-wasm-test`): immortal
    // `alloc_zeroed`, the same contract as `malloc_typed`. Never freed.
    let layout = Layout::from_size_align(bytes, std::mem::align_of::<GcEntries<K, V>>())
        .expect("entries layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let entries = raw as *mut GcEntries<K, V>;
    unsafe {
        (*entries).length = n;
    }
    entries
}

/// rgc.py `copy_struct_item` / `CopyStructEntry.specialize_call`:
/// `getinteriorfield` then `setinteriorfield` per `DICTENTRY` field, in
/// `get_ll_dict` `entryfields` order (`key`, `f_valid`, `value`, `f_hash`).
/// A GC pointer field is `framework.py` `var_needs_set_transform` /
/// `var_needsgc` then `transform_generic_set` / `_set_into_gc_array_part`:
/// `write_barrier_from_array` then the store. [`GcRefOffsets`] empty means
/// the field is not a GC pointer, so that instantiation has no barrier.
/// `CopyStructEntry` inlines those ops into the caller.
#[inline(always)]
unsafe fn copy_struct_item<K: Copy + GcRefOffsets, V: Copy + GcRefOffsets>(
    source: *mut GcEntries<K, V>,
    dest: *mut GcEntries<K, V>,
    si: usize,
    di: usize,
) {
    let src = entries_item_ptr(source);
    let dst = entries_item_ptr(dest);
    unsafe {
        if !K::GC_REF_OFFSETS.is_empty()
            && crate::gc_hook::try_gc_owns_object(dest as crate::gc_hook::GCREF)
        {
            majit_gc::gc_write_barrier_from_array(majit_ir::GcRef(dest as usize), di);
        }
        (*dst.add(di)).key = (*src.add(si)).key;
        (*dst.add(di)).f_valid = (*src.add(si)).f_valid;
        if !V::GC_REF_OFFSETS.is_empty()
            && crate::gc_hook::try_gc_owns_object(dest as crate::gc_hook::GCREF)
        {
            majit_gc::gc_write_barrier_from_array(majit_ir::GcRef(dest as usize), di);
        }
        (*dst.add(di)).value = (*src.add(si)).value;
        (*dst.add(di)).f_hash = (*src.add(si)).f_hash;
    }
}

/// rgc.py `copy_item`: `ARRAY.OF` is the `odictentry` struct, so
/// `copy_struct_item`.
#[inline(always)]
unsafe fn copy_item<K: Copy + GcRefOffsets, V: Copy + GcRefOffsets>(
    source: *mut GcEntries<K, V>,
    dest: *mut GcEntries<K, V>,
    si: usize,
    di: usize,
) {
    unsafe {
        copy_struct_item(source, dest, si, di);
    }
}

/// `rgc.py ll_arraycopy` for the `DICTENTRYARRAY` (`GcEntries<K, V>`).
///
/// Upstream's `@specialize.ll()` gives every ARRAY its own `ll_arraycopy`
/// graph, and the `length <= 1` `copy_item` head ("Hack to ensure that we
/// get a proper effectinfo.write_descrs_arrays") is the `setinteriorfield`
/// writeanalyze reads that ARRAY's interior-field effects off.
/// [`crate::object_array::jit_ll_arraycopy`] copies items as `PyObjectRef`;
/// a `DICTENTRY` is a struct, so this graph does the write barrier and
/// memcpy itself.
///
/// # Safety
/// `source` and `dest` are live `GcEntries` and both ranges are in bounds.
pub unsafe fn ll_arraycopy<K: Copy + GcRefOffsets, V: Copy + GcRefOffsets>(
    source: *mut GcEntries<K, V>,
    dest: *mut GcEntries<K, V>,
    source_start: usize,
    dest_start: usize,
    length: usize,
) {
    if length <= 1 {
        if length == 1 {
            unsafe {
                copy_item(source, dest, source_start, dest_start);
            }
        }
        return;
    }
    let mut slowpath = false;
    if crate::gc_hook::try_gc_owns_object(dest as crate::gc_hook::GCREF) {
        if crate::gc_hook::try_gc_owns_object(source as crate::gc_hook::GCREF) {
            slowpath = !majit_gc::gc_writebarrier_before_copy(
                majit_ir::GcRef(source as usize),
                majit_ir::GcRef(dest as usize),
                source_start,
                dest_start,
                length,
            );
        } else {
            slowpath = true;
        }
    }
    if slowpath {
        let mut i = 0usize;
        while i < length {
            unsafe {
                copy_item(source, dest, source_start + i, dest_start + i);
            }
            i += 1;
        }
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            entries_item_ptr(source).add(source_start),
            entries_item_ptr(dest).add(dest_start),
            length,
        );
    }
}

#[cfg(test)]
impl GcEntriesType for (u64, u64) {
    fn entries_gc_type_id() -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dummy_is_zero<T: EntryDummy + Copy>() {
        let value = T::dummy();
        let bytes = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };
        assert!(
            bytes.iter().all(|byte| *byte == 0),
            "{} dummy is not all-zero",
            std::any::type_name::<T>()
        );
    }

    #[test]
    fn entry_dummies_are_all_zero_bits() {
        assert_dummy_is_zero::<u64>();
        assert_dummy_is_zero::<i64>();
        assert_dummy_is_zero::<()>();
        assert_dummy_is_zero::<crate::pyobject::PyObjectRef>();
        assert_dummy_is_zero::<crate::dictmultiobject::ObjectKey>();
        assert_dummy_is_zero::<crate::dictmultiobject::BytesKey>();
        assert_dummy_is_zero::<crate::identitydict::IdentityKey>();
        assert_dummy_is_zero::<crate::setobject::IdentitySetKey>();
        assert_dummy_is_zero::<crate::celldict::StrKey>();
    }

    #[test]
    fn fresh_entries_are_zero_filled() {
        let entries = alloc_entries::<u64, u64>(4);
        assert!(!entries.is_null());
        unsafe {
            assert_eq!((*entries).length, 4);
            let item = entries_item_ptr(entries);
            for i in 0..4 {
                let entry = &*item.add(i);
                assert!(!entry.f_valid);
                assert_eq!(entry.key, 0);
                assert_eq!(entry.value, 0);
                assert_eq!(entry.f_hash, 0);
            }
        }
    }

    #[test]
    fn allocate_array_struct_at_typed_uses_the_published_entries_tid() {
        type K = i64;
        type V = crate::pyobject::PyObjectRef;
        let items_base = std::mem::offset_of!(GcEntries<K, V>, items);
        let item_size = std::mem::size_of::<Entry<K, V>>();
        let published = i64_pyobject_entries_gc_type_id();
        if majit_gc::gc_allocator_installed() {
            assert!(!majit_ir::descr::array_tid_is_unresolved(published));
            let arr = crate::object_array::allocate_array_struct_at_typed(
                2, item_size, items_base, published,
            );
            assert!(!arr.is_null());
            assert_eq!(crate::object_array::gcarray_len(arr), 2);
        } else {
            let arr = crate::object_array::allocate_array_struct_at_typed(
                2, item_size, items_base, published,
            );
            assert!(!arr.is_null());
            assert_eq!(crate::object_array::gcarray_len(arr), 2);
        }
    }

    #[test]
    fn object_key_entry_offsets_match_the_field_layout() {
        type Key = crate::dictmultiobject::ObjectKey;
        type Value = crate::pyobject::PyObjectRef;
        let key_base = std::mem::offset_of!(Entry<Key, Value>, key);
        let value_base = std::mem::offset_of!(Entry<Key, Value>, value);
        assert_eq!(
            entry_gc_ref_offsets::<Key, Value>(),
            vec![key_base + std::mem::offset_of!(Key, obj), value_base + 0,]
        );

        let unit_key = std::mem::offset_of!(Entry<Key, ()>, key);
        assert_eq!(
            entry_gc_ref_offsets::<Key, ()>(),
            vec![unit_key + std::mem::offset_of!(Key, obj)]
        );
        assert!(<()>::GC_REF_OFFSETS.is_empty());
    }

    #[test]
    fn ll_arraycopy_length_one_copies_each_entry_field() {
        let source = alloc_entries::<u64, u64>(2);
        let dest = alloc_entries::<u64, u64>(2);
        unsafe {
            let src = entries_item_ptr(source);
            (*src.add(1)).key = 11;
            (*src.add(1)).f_valid = true;
            (*src.add(1)).value = 22;
            (*src.add(1)).f_hash = 33;
            ll_arraycopy(source, dest, 1, 0, 1);
            let dst = entries_item_ptr(dest);
            assert_eq!((*dst).key, 11);
            assert!((*dst).f_valid);
            assert_eq!((*dst).value, 22);
            assert_eq!((*dst).f_hash, 33);
            assert!(!(*dst.add(1)).f_valid);
            assert_eq!((*dst.add(1)).key, 0);
            assert_eq!((*dst.add(1)).value, 0);
            assert_eq!((*dst.add(1)).f_hash, 0);
        }
    }

    #[test]
    fn ll_arraycopy_length_zero_is_a_no_op() {
        let source = alloc_entries::<u64, u64>(1);
        let dest = alloc_entries::<u64, u64>(1);
        unsafe {
            (*entries_item_ptr(source)).key = 7;
            (*entries_item_ptr(source)).f_valid = true;
            (*entries_item_ptr(source)).value = 8;
            (*entries_item_ptr(source)).f_hash = 9;
            ll_arraycopy(source, dest, 0, 0, 0);
            let dst = &*entries_item_ptr(dest);
            assert!(!dst.f_valid);
            assert_eq!(dst.key, 0);
            assert_eq!(dst.value, 0);
            assert_eq!(dst.f_hash, 0);
        }
    }

    #[test]
    fn ll_arraycopy_copies_a_prefix() {
        let source = alloc_entries::<u64, u64>(3);
        let dest = alloc_entries::<u64, u64>(3);
        unsafe {
            let src = entries_item_ptr(source);
            for i in 0..3 {
                (*src.add(i)).key = 10 + i as u64;
                (*src.add(i)).f_valid = true;
                (*src.add(i)).value = 20 + i as u64;
                (*src.add(i)).f_hash = 30 + i as u64;
            }
            ll_arraycopy(source, dest, 0, 0, 3);
            let dst = entries_item_ptr(dest);
            for i in 0..3 {
                let entry = &*dst.add(i);
                assert_eq!(entry.key, 10 + i as u64);
                assert!(entry.f_valid);
                assert_eq!(entry.value, 20 + i as u64);
                assert_eq!(entry.f_hash, 30 + i as u64);
            }
        }
    }
}
