use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bytesobject::BytesBlock;
use crate::object_array::{
    ItemsBlock, alloc_list_items_block_gc, dealloc_list_items_block, items_block_capacity,
    items_block_items_base,
};
use crate::pyobject::PyObjectRef;

/// PyPy `BytesListStrategy`'s erased `[rpython str]` storage.
///
/// Each entry is the GC pointer to a `BytesBlock`, not a boxed
/// `W_BytesObject`.  `ItemsBlock` is the runtime's `GcArray(GCREF)` shape, so
/// its existing varsize trace forwards both the backing block and every raw
/// string pointer it contains.
#[repr(C)]
pub struct BytesArray {
    pub block: *mut ItemsBlock,
    /// Live length (rlist.py `("length", Signed)`), read WITHOUT a lock.
    ///
    /// `Include/cpython/listobject.h PyList_GET_SIZE` answers a length under
    /// `Py_GIL_DISABLED` as `_Py_atomic_load_ssize_relaxed(&ob_size)` — a
    /// relaxed atomic load, no critical section — so a reader is entitled to a
    /// value from either side of a concurrent mutation but never to a torn one.
    /// A compiled trace reads this slot at a raw offset, through the
    /// length descriptor addressed by the matching `*_ARRAY_LEN_OFFSET`, which
    /// is why it cannot be a plain `usize`: the methods below write it while a
    /// compiled loop is reading it, and only an atomic makes that pair defined.
    /// Every write here is a relaxed store for the same reason — `&mut self`
    /// bounds no raw-pointer reader, so `get_mut()` would put the plain store
    /// straight back.
    ///
    /// Same size and bit validity as `usize`, so `offset_of!` and the JIT's
    /// `Type::Int` read are unchanged.
    pub(crate) len: AtomicUsize,
}

pub const BYTES_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(BytesArray, block);
pub const BYTES_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(BytesArray, len);

impl BytesArray {
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

    #[inline]
    fn base(&self) -> *mut PyObjectRef {
        unsafe { items_block_items_base(self.block) }
    }

    #[inline]
    pub fn empty() -> Self {
        Self {
            block: std::ptr::null_mut(),
            len: crate::object_array::length_cell(0),
        }
    }

    pub fn from_vec(values: Vec<*const BytesBlock>) -> Self {
        let mut refs = Vec::with_capacity(values.len());
        for value in values {
            refs.push(value as PyObjectRef);
        }
        let len = refs.len();
        Self {
            block: unsafe { alloc_list_items_block_gc(&refs) },
            len: crate::object_array::length_cell(len),
        }
    }

    /// `AbstractUnwrappedStrategy.get_empty_storage(sizehint)` for bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::empty();
        }
        Self {
            block: unsafe {
                crate::object_array::grow_list_items_block_gc(std::ptr::null_mut(), capacity, 0)
            },
            len: crate::object_array::length_cell(0),
        }
    }

    #[must_use]
    pub fn pin_block(&self) -> usize {
        let slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(self.block as PyObjectRef);
        slot
    }

    pub fn reload_block(&mut self, slot: usize) {
        self.block = crate::gc_roots::shadow_stack_get(slot) as *mut ItemsBlock;
    }

    pub fn install(&mut self, fresh: BytesArray) {
        let _roots = crate::gc_roots::push_roots();
        let slot = fresh.pin_block();
        *self = fresh;
        self.reload_block(slot);
    }

    #[inline]
    fn capacity(&self) -> usize {
        unsafe { items_block_capacity(self.block) }
    }

    #[inline]
    pub fn spare_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.len_relaxed())
    }

    #[inline]
    pub fn heap_capacity(&self) -> usize {
        self.capacity()
    }

    #[inline]
    pub fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity());
        self.set_len_relaxed(new_len);
    }

    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    /// The room `capacity` must already hold for `additional` more entries.
    ///
    /// A fresh block is young, and this array is embedded in the owning
    /// `W_ListObject` — the only object through which a collection reaches it.
    /// An old-gen owner that gains that edge without being on the remembered
    /// set is skipped by the minor collection that would forward the block, and
    /// the block, along with every `BytesBlock` reachable only through it, is
    /// reclaimed while the list still names it. `BytesArray` cannot reach its
    /// owner to barrier it, so it never allocates a block: the list reserves
    /// room through `W_ListObject::bytes_grow`, which barriers on both sides of
    /// the allocation. Refuse loudly rather than grow behind the owner's back.
    #[inline]
    fn assert_room(&self, additional: usize) {
        assert!(
            self.len_relaxed() + additional <= self.capacity(),
            "BytesArray needs {additional} more slot(s) than its capacity {}; \
             reserve through W_ListObject::bytes_grow first",
            self.capacity(),
        );
    }

    #[inline]
    fn barrier(&self) {
        if !self.block.is_null() {
            crate::gc_hook::try_gc_write_barrier(self.block as crate::gc_hook::GCREF);
        }
    }

    /// Generalize the block's cards before its items are permuted in place.
    ///
    /// A move stores no new reference, so [`Self::barrier`] does not answer for
    /// it: the referenced set is unchanged and only which card page holds each
    /// pointer moves. A minor reaches a carded array through its dirty pages
    /// alone, so a young pointer shifted into a clean page would not be
    /// scanned. The list side spells this `list_before_move_barrier`.
    #[inline]
    fn before_move_barrier(&self) {
        if !self.block.is_null() {
            crate::gc_hook::try_gc_write_barrier_before_move(self.block as crate::gc_hook::GCREF);
        }
    }

    pub fn push(&mut self, value: *const BytesBlock) {
        let _roots = crate::gc_roots::push_roots();
        let value_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value as PyObjectRef);
        self.assert_room(1);
        self.barrier();
        unsafe {
            *self.base().add(self.len_relaxed()) = crate::gc_roots::shadow_stack_get(value_slot)
        };
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

    pub fn as_slice(&self) -> &[*const BytesBlock] {
        unsafe {
            std::slice::from_raw_parts(self.base() as *const *const BytesBlock, self.len_relaxed())
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [*const BytesBlock] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.base() as *mut *const BytesBlock,
                self.len_relaxed(),
            )
        }
    }

    pub fn to_vec(&self) -> Vec<*const BytesBlock> {
        self.as_slice().to_vec()
    }

    pub fn insert(&mut self, index: usize, value: *const BytesBlock) {
        assert!(index <= self.len_relaxed());
        let _roots = crate::gc_roots::push_roots();
        let value_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value as PyObjectRef);
        self.assert_room(1);
        self.barrier();
        self.before_move_barrier();
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p, p.add(1), self.len_relaxed() - index);
            *p = crate::gc_roots::shadow_stack_get(value_slot);
        }
        self.set_len_relaxed(self.len_relaxed() + 1);
    }

    pub fn set(&mut self, index: usize, value: *const BytesBlock) {
        assert!(index < self.len_relaxed());
        let _roots = crate::gc_roots::push_roots();
        let slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value as PyObjectRef);
        self.barrier();
        unsafe { *self.base().add(index) = crate::gc_roots::shadow_stack_get(slot) };
    }

    pub fn remove(&mut self, index: usize) -> *const BytesBlock {
        assert!(index < self.len_relaxed());
        let value = self.as_slice()[index];
        self.before_move_barrier();
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p.add(1), p, self.len_relaxed() - index - 1);
            *p.add(self.len_relaxed() - index - 1) = std::ptr::null_mut();
        }
        self.set_len_relaxed(self.len_relaxed() - 1);
        value
    }

    pub fn pop(&mut self) -> *const BytesBlock {
        assert!(self.len_relaxed() > 0);
        let value = self.as_slice()[self.len_relaxed() - 1];
        self.set_len_relaxed(self.len_relaxed() - 1);
        unsafe { *self.base().add(self.len_relaxed()) = std::ptr::null_mut() };
        value
    }

    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    pub fn splice(&mut self, start: usize, remove_count: usize, values: &[*const BytesBlock]) {
        let old_len = self.len_relaxed();
        let start = start.min(old_len);
        let removed = remove_count.min(old_len - start);
        let new_len = old_len - removed + values.len();
        let _roots = crate::gc_roots::push_roots();
        let root_base = crate::gc_roots::shadow_stack_len();
        for &value in values {
            let _ = crate::gc_roots::pin_root(value as PyObjectRef);
        }
        assert!(
            new_len <= self.capacity(),
            "BytesArray splice needs {new_len} slots but capacity is {}; \
             reserve through W_ListObject::bytes_grow first",
            self.capacity(),
        );
        self.barrier();
        self.before_move_barrier();
        unsafe {
            let base = self.base();
            std::ptr::copy(
                base.add(start + removed),
                base.add(start + values.len()),
                old_len - start - removed,
            );
            self.set_len_relaxed(new_len);
            for i in 0..values.len() {
                *base.add(start + i) = crate::gc_roots::shadow_stack_get(root_base + i);
            }
            for i in new_len..old_len {
                *base.add(i) = std::ptr::null_mut();
            }
        }
    }

    pub fn drain(&mut self, range: std::ops::Range<usize>) {
        assert!(range.start <= range.end && range.end <= self.len_relaxed());
        let count = range.end - range.start;
        if count == 0 {
            return;
        }
        self.before_move_barrier();
        unsafe {
            let base = self.base();
            std::ptr::copy(
                base.add(range.end),
                base.add(range.start),
                self.len_relaxed() - range.end,
            );
            for i in self.len_relaxed() - count..self.len_relaxed() {
                *base.add(i) = std::ptr::null_mut();
            }
        }
        self.set_len_relaxed(self.len_relaxed() - count);
    }

    pub fn clear(&mut self) {
        unsafe {
            for i in 0..self.len_relaxed() {
                *self.base().add(i) = std::ptr::null_mut();
            }
        }
        self.set_len_relaxed(0);
    }
}

impl Drop for BytesArray {
    fn drop(&mut self) {
        unsafe { dealloc_list_items_block(self.block) };
    }
}

impl Index<usize> for BytesArray {
    type Output = *const BytesBlock;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for BytesArray {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}
