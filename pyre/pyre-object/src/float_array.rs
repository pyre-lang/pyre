use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::object_array::{
    TYPED_ITEMS_BLOCK_ITEMS_OFFSET, TypedItemsBlock, alloc_typed_items_block,
    dealloc_typed_items_block, gc_float_array_gc_type_id, grow_float_items_block,
    try_alloc_typed_items_block, typed_items_block_capacity,
};

pub const FLOAT_ARRAY_INLINE_CAP: usize = 4;

/// Unboxed `float` list storage — `listobject.py` FloatListStrategy
/// `lstorage = erase([float])`, i.e. a `Ptr(GcArray(Float))`.
///
/// `rlist.py:116` `LIST = GcStruct("list", ("length", Signed), ("items",
/// Ptr(GcArray(item))))`: the live length is `len` and the items array is the
/// length-prefixed [`TypedItemsBlock`] (`[capacity][f64...]`) reached through
/// `block`. The items base and allocated capacity are read from `block` on
/// demand — there is no cached interior pointer, so the JIT addresses the array
/// as a GC ref (`GetfieldGcR(block) → GetarrayitemGcF`) the gcmap relocates.
#[repr(C)]
pub struct FloatArray {
    /// `Ptr(GcArray(Float))` — the backing block (`l.items`). Null in the empty
    /// form ([`FloatArray::empty`]), where the live length and the allocated
    /// capacity are both zero.
    pub block: *mut TypedItemsBlock,
    /// Live length (rlist.py `("length", Signed)`), read WITHOUT a lock.
    ///
    /// `Include/cpython/listobject.h PyList_GET_SIZE` answers a length under
    /// `Py_GIL_DISABLED` as `_Py_atomic_load_ssize_relaxed(&ob_size)` — a
    /// relaxed atomic load, no critical section — so a reader is entitled to a
    /// value from either side of a concurrent mutation but never to a torn one.
    /// The JIT's `len` fold lowers to exactly that load, addressing this slot
    /// by [`FLOAT_ARRAY_LEN_OFFSET`], which is why it cannot be a plain
    /// `usize`: the methods below write it while a compiled loop is reading it,
    /// and only an atomic makes that pair defined.  Every write here is a
    /// relaxed store for the same reason — `&mut self` bounds no raw-pointer
    /// reader, so `get_mut()` would put the plain store straight back.
    ///
    /// Same size and bit validity as `usize`, so `offset_of!` and the JIT's
    /// `Type::Int` read are unchanged.
    len: AtomicUsize,
}

pub const FLOAT_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(FloatArray, block);
pub const FLOAT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(FloatArray, len);

impl FloatArray {
    /// `_Py_atomic_load_ssize_relaxed(&ob_size)`.
    #[inline]
    fn len_relaxed(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// `_Py_atomic_store_ssize_relaxed(&ob_size, n)`.
    #[inline]
    fn set_len_relaxed(&self, n: usize) {
        self.len.store(n, Ordering::Relaxed);
    }

    /// Items base pointer (`&l.items[0]`), derived from `block`. The
    /// [`crate::int_array::IntArray::base`] twin — see it for why the empty
    /// form's null `block` is offset rather than branched on.
    #[inline]
    fn base(&self) -> *mut f64 {
        (self.block as *mut u8).wrapping_add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET) as *mut f64
    }

    /// Storage for a list whose strategy does not read this array. The
    /// [`crate::int_array::IntArray::empty`] twin — see it for why
    /// `from_vec(Vec::new())` is not the same thing.
    #[inline]
    pub fn empty() -> Self {
        Self {
            block: std::ptr::null_mut(),
            len: crate::object_array::length_cell(0),
        }
    }

    pub fn from_vec(values: Vec<f64>) -> Self {
        let len = values.len();
        let arr = Self {
            block: unsafe { alloc_typed_items_block(len, gc_float_array_gc_type_id()) },
            len: crate::object_array::length_cell(len),
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), arr.base(), len);
        }
        arr
    }

    /// `ll_mul` / `ll_mul_loop`: `erase(l * times)` into one items array.
    /// `length == 1` is `ll_alloc_and_set`; otherwise the non-gcptr doubling
    /// copy (`f64` items hold no GC pointers).
    pub fn from_repeated(src: &[f64], times: usize) -> Option<Self> {
        let length = src.len();
        let resultlen = length.checked_mul(times)?;
        if resultlen == 0 {
            return Some(Self::empty());
        }
        let arr = Self {
            block: unsafe { try_alloc_typed_items_block(resultlen, gc_float_array_gc_type_id())? },
            len: crate::object_array::length_cell(resultlen),
        };
        unsafe {
            if length == 1 {
                std::slice::from_raw_parts_mut(arr.base(), resultlen).fill(src[0]);
            } else {
                std::ptr::copy_nonoverlapping(src.as_ptr(), arr.base(), length);
                let mut j = length;
                while j < resultlen {
                    let copy_length = std::cmp::min(resultlen - j, j);
                    // `copy_length <= j`, so `res[0..copy_length]` and
                    // `res[j..j + copy_length]` are disjoint.
                    std::ptr::copy_nonoverlapping(arr.base(), arr.base().add(j), copy_length);
                    j += copy_length;
                }
                debug_assert_eq!(j, resultlen);
            }
        }
        Some(arr)
    }

    /// `AbstractUnwrappedStrategy.get_empty_storage(sizehint)` for floats.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::empty();
        }
        Self {
            block: unsafe { alloc_typed_items_block(capacity, gc_float_array_gc_type_id()) },
            len: crate::object_array::length_cell(0),
        }
    }

    /// Pin `block` on the shadow stack and return its slot. The
    /// [`crate::int_array::IntArray::pin_block`] twin — see it for why an
    /// old-gen, non-moving block still has to be rooted before the next GC
    /// operation.
    #[must_use]
    pub fn pin_block(&self) -> usize {
        let slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(self.block as crate::PyObjectRef);
        slot
    }

    /// Re-read `block` from the slot [`Self::pin_block`] returned — the
    /// `pop_roots` half of the bracket.
    pub fn reload_block(&mut self, slot: usize) {
        self.block = crate::gc_roots::shadow_stack_get(slot) as *mut TypedItemsBlock;
    }

    /// Replace the storage in place, keeping the incoming block rooted across
    /// the teardown of the outgoing one. The
    /// [`crate::int_array::IntArray::install`] twin — see it for the sweep
    /// window a bare `*self = fresh` leaves open.
    pub fn install(&mut self, fresh: FloatArray) {
        let _roots = crate::gc_roots::push_roots();
        let slot = fresh.pin_block();
        *self = fresh;
        self.reload_block(slot);
    }

    /// Allocated capacity (`len(l.items)`, rlist.py:251), read from the block
    /// header.
    #[inline]
    fn capacity(&self) -> usize {
        unsafe { typed_items_block_capacity(self.block) }
    }

    #[inline]
    pub fn spare_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.len_relaxed())
    }

    /// Allocated capacity (block header). The no-resize append fast path
    /// guards `len < heap_capacity()`, mirroring `_ll_list_resize_ge`
    /// (rlist.py:285).
    #[inline]
    pub fn heap_capacity(&self) -> usize {
        self.capacity()
    }

    /// Store the live length without touching the block. The caller must
    /// guarantee `new_len <= heap_capacity()` (the no-resize precondition);
    /// mirrors `_ll_list_resize_ge`'s `l.length = newsize` (rlist.py).
    /// Enforced here because this is safe/public: a `len` past the allocated
    /// capacity would make `as_slice`/`as_mut_slice` build out-of-bounds
    /// slices (UB).
    #[inline]
    pub fn set_len(&mut self, new_len: usize) {
        let cap = self.capacity();
        assert!(
            new_len <= cap,
            "FloatArray::set_len precondition violated: new_len ({new_len}) > capacity ({cap})"
        );
        self.set_len_relaxed(new_len);
    }

    /// Float list storage is always a separate block (no inline buffer);
    /// upstream `erase([float])` has no inline bit either.
    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    /// `_ll_list_resize_hint_really(overallocate=True)` for this block.
    pub fn grow(&mut self, min_cap: usize) {
        let extra = if min_cap < 9 { 3 } else { 6 };
        let target_cap = min_cap
            .saturating_add(extra)
            .saturating_add(min_cap >> 3)
            .max(FLOAT_ARRAY_INLINE_CAP);
        self.block = unsafe { grow_float_items_block(self.block, target_cap, self.len_relaxed()) };
    }

    pub fn push(&mut self, value: f64) {
        if self.len_relaxed() == self.capacity() {
            self.grow(self.len_relaxed() + 1);
        }
        unsafe {
            *self.base().add(self.len_relaxed()) = value;
        }
        self.set_len_relaxed(self.len_relaxed() + 1);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len_relaxed()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_relaxed() == 0
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.as_slice().to_vec()
    }

    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.base(), self.len_relaxed()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.base(), self.len_relaxed()) }
    }

    pub fn insert(&mut self, index: usize, value: f64) {
        assert!(index <= self.len_relaxed());
        if self.len_relaxed() == self.capacity() {
            self.grow(self.len_relaxed() + 1);
        }
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p, p.add(1), self.len_relaxed() - index);
            *p = value;
        }
        self.set_len_relaxed(self.len_relaxed() + 1);
    }

    pub fn remove(&mut self, index: usize) -> f64 {
        assert!(index < self.len_relaxed());
        let value = self.as_slice()[index];
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p.add(1), p, self.len_relaxed() - index - 1);
        }
        self.set_len_relaxed(self.len_relaxed() - 1);
        value
    }

    pub fn pop(&mut self) -> f64 {
        assert!(self.len_relaxed() > 0);
        let value = self.as_slice()[self.len_relaxed() - 1];
        self.set_len_relaxed(self.len_relaxed() - 1);
        value
    }

    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    pub fn splice(&mut self, start: usize, remove_count: usize, new_values: &[f64]) {
        let old_len = self.len_relaxed();
        let s = start.min(old_len);
        let slicelength = remove_count.min(old_len - s);
        let len2 = new_values.len();
        let new_len = old_len - slicelength + len2;
        if len2 > slicelength {
            if new_len > self.capacity() {
                self.grow(new_len);
            }
            unsafe {
                let base = self.base();
                std::ptr::copy(
                    base.add(s + slicelength),
                    base.add(s + len2),
                    old_len - s - slicelength,
                );
            }
            self.set_len_relaxed(new_len);
        } else if slicelength > len2 {
            unsafe {
                let base = self.base();
                std::ptr::copy(
                    base.add(s + slicelength),
                    base.add(s + len2),
                    old_len - s - slicelength,
                );
            }
            self.set_len_relaxed(new_len);
        }
        if len2 > 0 {
            self.as_mut_slice()[s..s + len2].copy_from_slice(new_values);
        }
    }

    pub fn drain(&mut self, range: std::ops::Range<usize>) {
        let start = range.start;
        let end = range.end;
        assert!(start <= end && end <= self.len_relaxed());
        let count = end - start;
        if count == 0 {
            return;
        }
        unsafe {
            let p = self.base().add(start);
            std::ptr::copy(p.add(count), p, self.len_relaxed() - end);
        }
        self.set_len_relaxed(self.len_relaxed() - count);
    }

    pub fn clear(&mut self) {
        self.set_len_relaxed(0);
    }
}

impl Drop for FloatArray {
    fn drop(&mut self) {
        unsafe {
            dealloc_typed_items_block(self.block);
        }
    }
}

impl Index<usize> for FloatArray {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.base().add(index) }
    }
}

impl IndexMut<usize> for FloatArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.base().add(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_repeated_matches_naive_repeat() {
        for length in [2usize, 3] {
            let src: Vec<f64> = (0..length as i64).map(|n| n as f64).collect();
            for times in [0usize, 1, 2, 5, 7] {
                let expected: Vec<f64> = src.repeat(times);
                let arr = FloatArray::from_repeated(&src, times).unwrap();
                assert_eq!(
                    arr.as_slice(),
                    expected.as_slice(),
                    "length={length} times={times}"
                );
            }
        }
    }
}
