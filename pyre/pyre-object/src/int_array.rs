use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::object_array::{
    TYPED_ITEMS_BLOCK_ITEMS_OFFSET, TypedItemsBlock, alloc_typed_items_block,
    dealloc_typed_items_block, gc_int_array_gc_type_id, grow_int_items_block,
    try_alloc_typed_items_block, typed_items_block_capacity,
};

/// Small-buffer capacity constant retained for the append/pop inline-capacity
/// trace path (`is_inline()` is always false, so it is never consulted at
/// runtime).
pub const INT_ARRAY_INLINE_CAP: usize = 4;

/// Unboxed `int` list storage — `listobject.py` IntegerListStrategy
/// `lstorage = erase([int])`, i.e. a `Ptr(GcArray(Signed))`.
///
/// `rlist.py:116` `LIST = GcStruct("list", ("length", Signed), ("items",
/// Ptr(GcArray(item))))`: the live length is `len` and the items array is the
/// length-prefixed [`TypedItemsBlock`] (`[capacity][i64...]`) reached through
/// `block`. The items base and allocated capacity are read from `block` on
/// demand (`len(l.items)` = the block's capacity header) — there is no cached
/// interior pointer, so the JIT can address the array as a GC ref
/// (`GetfieldGcR(block) → GetarrayitemGcI`) that the gcmap relocates on a move.
#[repr(C)]
pub struct IntArray {
    /// `Ptr(GcArray(Signed))` — the backing block (`l.items`). Null in the
    /// empty form ([`IntArray::empty`]), where the live length and the
    /// allocated capacity are both zero.
    pub block: *mut TypedItemsBlock,
    /// Live length (rlist.py `("length", Signed)`), read WITHOUT a lock.
    ///
    /// `Include/cpython/listobject.h PyList_GET_SIZE` answers a length under
    /// `Py_GIL_DISABLED` as `_Py_atomic_load_ssize_relaxed(&ob_size)` — a
    /// relaxed atomic load, no critical section — so a reader is entitled to a
    /// value from either side of a concurrent mutation but never to a torn one.
    /// The JIT's `len` fold lowers to exactly that load, addressing this slot
    /// by [`INT_ARRAY_LEN_OFFSET`], which is why it cannot be a plain
    /// `usize`: the methods below write it while a compiled loop is reading it,
    /// and only an atomic makes that pair defined.  Every write here is a
    /// relaxed store for the same reason — `&mut self` bounds no raw-pointer
    /// reader, so `get_mut()` would put the plain store straight back.
    ///
    /// Same size and bit validity as `usize`, so `offset_of!` and the JIT's
    /// `Type::Int` read are unchanged.
    len: AtomicUsize,
}

pub const INT_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(IntArray, block);
pub const INT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(IntArray, len);

impl IntArray {
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

    /// Items base pointer (`&l.items[0]`), derived from `block`.
    ///
    /// `wrapping_add`, so the empty form's null `block` yields the items offset
    /// itself — a non-null, 8-aligned address `as_slice` may hand to
    /// `from_raw_parts` at length zero. No other caller reaches it: the empty
    /// form's capacity is zero, so every write goes through [`Self::grow`]
    /// first.
    #[inline]
    fn base(&self) -> *mut i64 {
        (self.block as *mut u8).wrapping_add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET) as *mut i64
    }

    /// Storage for a list whose strategy does not read this array: no block, no
    /// live length, no allocated capacity.
    ///
    /// This is the shape traced code already builds — `emit_typed_list_inline`
    /// writes one typed pair and leaves the other's `block` NULL from
    /// `NewWithVtable`'s zero fill. `from_vec(Vec::new())` does not produce it:
    /// `try_alloc_typed_items_block` clamps `cap` to 1 (rlist.py:251
    /// overallocation) and takes the old-gen `try_gc_alloc_stable_raw`, so each
    /// non-Integer list bought a block it never reads.
    ///
    /// The block helpers already read null as capacity zero, `grow` and
    /// `dealloc` already special-case it, and `list_object_custom_trace`
    /// forwards the owning slot only when the collector owns what it holds.
    #[inline]
    pub fn empty() -> Self {
        Self {
            block: std::ptr::null_mut(),
            len: crate::object_array::length_cell(0),
        }
    }

    pub fn from_vec(values: Vec<i64>) -> Self {
        let len = values.len();
        let arr = Self {
            block: unsafe { alloc_typed_items_block(len, gc_int_array_gc_type_id()) },
            len: crate::object_array::length_cell(len),
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), arr.base(), len);
        }
        arr
    }

    /// `ll_mul` / `ll_mul_loop`: `erase(l * times)` into one items array.
    /// `length == 1` is `ll_alloc_and_set`; otherwise the non-gcptr doubling
    /// copy (`i64` items hold no GC pointers).
    pub fn from_repeated(src: &[i64], times: usize) -> Option<Self> {
        let length = src.len();
        let resultlen = length.checked_mul(times)?;
        if resultlen == 0 {
            return Some(Self::empty());
        }
        let arr = Self {
            block: unsafe { try_alloc_typed_items_block(resultlen, gc_int_array_gc_type_id())? },
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

    /// `AbstractUnwrappedStrategy.get_empty_storage(sizehint)`: allocate the
    /// exact hinted RPython items array while keeping its live length zero.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::empty();
        }
        Self {
            block: unsafe { alloc_typed_items_block(capacity, gc_int_array_gc_type_id()) },
            len: crate::object_array::length_cell(0),
        }
    }

    /// `rlist.py ll_newlist`: a Signed items array whose live length is `len`.
    /// `ll_listslice_startstop` fills it with `ll_arraycopy`; the payload is
    /// uninitialized until that copy.
    pub fn with_len(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        Self {
            block: unsafe { alloc_typed_items_block(len, gc_int_array_gc_type_id()) },
            len: crate::object_array::length_cell(len),
        }
    }

    /// Adopt an already-allocated Signed items block. The block is the
    /// `ll_newlist` result; `ll_arraycopy` fills it before the list header
    /// takes the edge.
    #[inline]
    pub fn from_block(block: *mut TypedItemsBlock, len: usize) -> Self {
        Self {
            block,
            len: crate::object_array::length_cell(len),
        }
    }

    /// Pin `block` on the shadow stack and return its slot, so the block stays
    /// live across a following GC operation.
    ///
    /// `gct_fv_gc_malloc` (`framework.py`) brackets every malloc with
    /// `push_roots`/`pop_roots` over the live vars, and that bracket is about
    /// *liveness*, not only relocation: old-gen here is mark-sweep
    /// (`incminimark.py` `STATE_SWEEPING` frees every old object that did not
    /// gain `GCFLAG_VISITED` during this cycle), so **non-moving does not mean
    /// non-collected**. A block born while the cycle is still scanning gets no
    /// `VISITED` bit (`collector.rs` `oldgen_birth_flags`), and until the owning
    /// field store lands it is reachable from a Rust local only — no root, no
    /// heap edge — so this cycle's sweep is entitled to reclaim it.
    ///
    /// Every GC operation in that window is a real safepoint: with more than one
    /// registered thread `gc_op` takes `gc_op_slow`, which clears `running` and
    /// blocks on the GC mutex, letting another mutator drive a full cycle to
    /// completion. `try_gc_alloc_stable_raw` and the `try_gc_owns_object` query
    /// inside `dealloc_typed_items_block` both qualify.
    ///
    /// [`crate::gc_roots::pin_root`] publishes the raw value to the shadow stack
    /// before it performs its own forwarding query, so no safepoint sits between
    /// the allocation and the root becoming visible.
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
    /// the teardown of the outgoing one.
    ///
    /// `*self = fresh` alone is unsound here: Rust evaluates the right-hand side
    /// (which allocates `fresh.block`), then drops the value being replaced, and
    /// [`Drop`] runs `dealloc_typed_items_block` whose `try_gc_owns_object` query
    /// is a safepoint — with `fresh.block` still unrooted and unreferenced. A
    /// concurrent cycle reaching `STATE_SWEEPING` in that window frees the cell
    /// into the arena free list, and the store below then publishes freed memory
    /// as the list's storage. See [`Self::pin_block`] for why old-gen membership
    /// is no protection.
    pub fn install(&mut self, fresh: IntArray) {
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
    /// guards `len < heap_capacity()` before writing past the live length,
    /// mirroring `_ll_list_resize_ge`'s `len(items) >= length + 1` check
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
            "IntArray::set_len precondition violated: new_len ({new_len}) > capacity ({cap})"
        );
        self.set_len_relaxed(new_len);
    }

    /// Integer list storage is always a separate block (no inline buffer).
    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    /// `_ll_list_resize_hint_really(overallocate=True)` for this block.
    pub fn grow(&mut self, min_cap: usize) {
        // rlist.py `_ll_list_resize_hint_really(overallocate=True)`:
        // 0, 4, 8, 16, 25, 35, ...
        let extra = if min_cap < 9 { 3 } else { 6 };
        let target_cap = min_cap
            .saturating_add(extra)
            .saturating_add(min_cap >> 3)
            .max(INT_ARRAY_INLINE_CAP);
        self.block = unsafe { grow_int_items_block(self.block, target_cap, self.len_relaxed()) };
    }

    pub fn push(&mut self, value: i64) {
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

    pub fn as_slice(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.base(), self.len_relaxed()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.base(), self.len_relaxed()) }
    }

    pub fn to_vec(&self) -> Vec<i64> {
        self.as_slice().to_vec()
    }

    pub fn insert(&mut self, index: usize, value: i64) {
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

    pub fn remove(&mut self, index: usize) -> i64 {
        assert!(index < self.len_relaxed());
        let value = self.as_slice()[index];
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p.add(1), p, self.len_relaxed() - index - 1);
        }
        self.set_len_relaxed(self.len_relaxed() - 1);
        value
    }

    pub fn pop(&mut self) -> i64 {
        assert!(self.len_relaxed() > 0);
        let value = self.as_slice()[self.len_relaxed() - 1];
        self.set_len_relaxed(self.len_relaxed() - 1);
        value
    }

    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    pub fn splice(&mut self, start: usize, remove_count: usize, new_values: &[i64]) {
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

impl Drop for IntArray {
    fn drop(&mut self) {
        unsafe {
            dealloc_typed_items_block(self.block);
        }
    }
}

impl Index<usize> for IntArray {
    type Output = i64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.base().add(index) }
    }
}

impl IndexMut<usize> for IntArray {
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
            let src: Vec<i64> = (0..length as i64).collect();
            for times in [0usize, 1, 2, 5, 7] {
                let expected: Vec<i64> = src.repeat(times);
                let arr = IntArray::from_repeated(&src, times).unwrap();
                assert_eq!(
                    arr.as_slice(),
                    expected.as_slice(),
                    "length={length} times={times}"
                );
            }
        }
    }
}
