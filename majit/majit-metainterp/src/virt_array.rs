//! GcArray-shaped backing for a virtualizable array field.
//!
//! `virtualizable.py:57-58` builds one `cpu.arraydescrof(...)` per virtualizable
//! array, and the field those descrs address is a `Ptr(GcArray)`: a single
//! pointer word in the virtualizable struct, aimed at a block whose length word
//! and payload sit at fixed offsets from the block address. That shape is what
//! lets `compile.py:441-457` rebuild every element of the array inside the
//! compiled entry from the virtualizable alone — `GETFIELD_GC_R` for the
//! pointer, then one `GETARRAYITEM_GC_*` per element, both expressible in trace
//! IR with nothing but byte offsets.
//!
//! A Rust `Vec<T>` embedded by value cannot serve that entry. Its data pointer
//! is one of three words in an order the language does not specify, so no field
//! load portably finds it, and reading the wrong word yields a capacity as a
//! base address. [`VirtArray`] is the same logical container with the upstream
//! physical shape: one owned pointer to a `[length][payload…]` block.
//!
//! The block is deliberately allocated *outside* any collector. A state struct
//! carrying these arrays is a stack-resident non-GC local — its identity as a
//! virtualizable is its own address (`virtualizable_heap_ptr`) — so nothing
//! traces it, and a block in a moving nursery reachable only from there would be
//! collected or relocated behind the array's back. This mirrors the off-GC
//! JITFRAME allocation in `majit_backend::jitframe::alloc_off_gc_jitframe`,
//! including its prefix: a size slot so the block pointer alone can free it, and
//! a zeroed header word in front so a header-relative write-barrier probe reads
//! in-bounds and answers "no barrier".

use std::alloc::Layout;
use std::marker::PhantomData;

use majit_ir::Type;

use crate::virtualizable::VirtualizableInfo;

/// Bytes reserved for the total-allocation-size slot in front of a block.
const BLOCK_SIZE_SLOT: usize = majit_gc::header::GcHeader::SIZE;
/// Bytes reserved for the zeroed header word in front of a block.
const BLOCK_HEADER: usize = majit_gc::header::GcHeader::SIZE;
/// Distance from the block pointer back to the size slot.
///
/// The size slot and the header word sit immediately behind the block pointer,
/// in that order, so a header-relative probe at a small negative offset lands
/// inside the allocation.
const BLOCK_PREFIX: usize = BLOCK_SIZE_SLOT + BLOCK_HEADER;
const _: () = assert!(BLOCK_SIZE_SLOT >= std::mem::size_of::<u64>());

/// Alignment of a block's allocation, for an item type of `align`.
///
/// The prefix holds a `u64` size slot and a `GcHeader`, and the block starts
/// with a `usize` length word, so the allocation must carry their alignments
/// alongside the item's — as alignments, not as sizes: a size is not required
/// to be a power of two, and `Layout` rejects one that is not.
const fn block_align(align: usize) -> usize {
    let mut result = align;
    if std::mem::align_of::<u64>() > result {
        result = std::mem::align_of::<u64>();
    }
    if std::mem::align_of::<usize>() > result {
        result = std::mem::align_of::<usize>();
    }
    if majit_gc::header::GcHeader::ALIGN > result {
        result = majit_gc::header::GcHeader::ALIGN;
    }
    result
}

/// Distance from the allocation base to the block pointer handed out, for an
/// item type of `align`.
///
/// The allocation base carries the item's alignment, but the block pointer is
/// the base advanced past the prefix, and the payload is the block pointer
/// advanced by [`block_items_offset`]. Neither of those two steps can repair an
/// alignment the first one broke, so the prefix — not the payload offset — is
/// where an item needing more alignment than the size slot and header word
/// provide is padded for. The padding goes in front of the size slot, leaving
/// the header word adjacent to the block.
///
/// ```text
///   base            block-16        block-8        block
///   [ padding … ]   [ total size ]  [ header ]     [ length ][ payload … ]
/// ```
const fn block_base_offset(align: usize) -> usize {
    let word = std::mem::align_of::<usize>();
    let align = if align < word { word } else { align };
    BLOCK_PREFIX.div_ceil(align) * align
}

/// Offset of the length word from the block pointer.
///
/// `rlist.py:251` puts the GcArray length first and the items straight after it;
/// `majit_rlib::lltypesystem::rlist::TypedItemsBlock` is the same lowering for
/// the list-strategy body.
pub const BLOCK_LENGTH_OFFSET: usize = 0;

/// Offset of item 0 from the block pointer, for an item type of `align`.
///
/// The length word is one machine word, but an item may need more alignment
/// than that (an `f64` payload on a 32-bit target), so the payload starts at the
/// length word's end rounded up to the item's alignment.
const fn block_items_offset(align: usize) -> usize {
    let word = std::mem::size_of::<usize>();
    let align = if align < word { word } else { align };
    word.div_ceil(align) * align
}

mod sealed {
    /// Sealed: only this module may name a `[..; virt]` backing.
    pub trait PointerShaped {}
}

/// A container a `[..; virt]` state field may be declared with.
///
/// The interpreter author picks the backing by choosing the field's Rust type;
/// this trait is how the generated `VirtualizableInfo` builder and the
/// fresh-entry state constructor stay written once. Every read and write of
/// the elements goes through the slice the container derefs to.
///
/// Sealed, because `LENGTH_OFFSET` and `ITEMS_OFFSET` are offsets from the
/// FIELD'S OWN WORD: the state field is registered as a direct pointer, and
/// the compiled entry and the blackhole both dereference that word to reach
/// the block. A backing that kept anything else there -- a length inline
/// ahead of its data, say -- would satisfy this trait's methods and still
/// hand those paths a number to dereference. Upstream has the one shape too
/// (`Ptr(GcArray)`), so a second backing belongs here, beside the paths that
/// read it, rather than downstream.
pub trait VirtArrayBacking:
    sealed::PointerShaped + std::ops::DerefMut<Target = [Self::Item]>
{
    /// Element type. Restricted to `Copy` so a block is released by freeing it,
    /// with no element drop glue to run.
    type Item: Copy;

    /// Offset of the length word from the field's pointer value.
    const LENGTH_OFFSET: usize;
    /// Offset of item 0 from the field's pointer value.
    const ITEMS_OFFSET: usize;

    /// `len` copies of `value` — the fresh-entry constructor.
    fn filled(value: Self::Item, len: usize) -> Self;
}

/// A `Vec`-like array whose storage is a single owned pointer to a
/// `[length][payload…]` block.
///
/// Read and write the elements through the slice this derefs to; the pointer
/// word is what makes the container addressable from compiled code, and nothing
/// else about it is meant to be visible to interpreter code.
///
/// Empty is still a real block, so the pointer is never null and item-0
/// addressing never offsets from null.
#[repr(C)]
pub struct VirtArray<T: Copy> {
    /// Points past the size slot and header word at the length word.
    block: *mut u8,
    _item: PhantomData<T>,
}

// The block is owned exclusively by this handle: no other handle can reach it,
// and dropping the handle frees it. So it may cross threads exactly when its
// items may.
unsafe impl<T: Copy + Send> Send for VirtArray<T> {}
unsafe impl<T: Copy + Sync> Sync for VirtArray<T> {}

const _: () = assert!(std::mem::size_of::<VirtArray<i64>>() == std::mem::size_of::<usize>());

impl<T: Copy> VirtArray<T> {
    /// Offset of the length word from the field's pointer value.
    pub const LENGTH_OFFSET: usize = BLOCK_LENGTH_OFFSET;
    /// Offset of item 0 from the field's pointer value.
    pub const ITEMS_OFFSET: usize = block_items_offset(std::mem::align_of::<T>());
    /// Distance from the allocation base to the field's pointer value.
    const BASE_OFFSET: usize = block_base_offset(std::mem::align_of::<T>());

    /// The payload lives at base + [`Self::BASE_OFFSET`] + [`Self::ITEMS_OFFSET`],
    /// and the slice handed out over it is only sound when that address carries
    /// the item's alignment. The base does, so the two offsets have to as well.
    const _PAYLOAD_ALIGNED: () =
        assert!((Self::BASE_OFFSET + Self::ITEMS_OFFSET).is_multiple_of(std::mem::align_of::<T>()));

    /// An empty array.
    pub fn new() -> Self {
        Self::with_len(0)
    }

    /// `len` items of whatever the allocation returned. Only useful when the
    /// caller writes every slot before reading it; [`VirtArray::filled`] is the
    /// initialized form.
    fn with_len(len: usize) -> Self {
        let block = Self::alloc_block(len);
        Self {
            block,
            _item: PhantomData,
        }
    }

    /// `len` copies of `value`.
    pub fn filled(value: T, len: usize) -> Self {
        let mut this = Self::with_len(len);
        this.fill(value);
        this
    }

    /// A copy of `items`.
    pub fn from_slice(items: &[T]) -> Self {
        let mut this = Self::with_len(items.len());
        this.copy_from_slice(items);
        this
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        unsafe { *(self.block.add(Self::LENGTH_OFFSET) as *const usize) }
    }

    /// Whether the array holds no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Grow or shrink to `new_len`, filling any new slot with `value` and
    /// keeping every slot both lengths have.
    ///
    /// A block has no spare capacity — its length word is the allocation's
    /// length, which is what makes the length readable from the block address
    /// alone — so a length change is a reallocation. A length that does not
    /// change is not, which is the case a caller reusing one array across calls
    /// is in.
    ///
    /// The unchanged case inlines to the length compare and the reallocation
    /// does not, so a caller that reuses one array across calls pays a load and
    /// a branch rather than a call frame.
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len == self.len() {
            return;
        }
        self.reallocate(new_len, value);
    }

    #[inline(never)]
    fn reallocate(&mut self, new_len: usize, value: T) {
        let old_len = self.len();
        let mut grown = Self::with_len(new_len);
        let kept = old_len.min(new_len);
        grown[..kept].copy_from_slice(&self[..kept]);
        for slot in &mut grown[kept..] {
            *slot = value;
        }
        *self = grown;
    }

    /// Discard every item, leaving a zero-length array.
    ///
    /// A block has no spare capacity, so this releases the old block and
    /// allocates an empty one; a caller counting allocations sees both. An
    /// array that is already zero-length keeps the block it has.
    pub fn clear(&mut self) {
        // `resize` needs an item to fill with and a shrink to zero never uses
        // one, so go straight to the empty block rather than invent a value.
        if !self.is_empty() {
            *self = Self::with_len(0);
        }
    }

    /// Allocate a block for `len` items, off any collector, and write its
    /// length word. The items are left as the allocator returned them.
    fn alloc_block(len: usize) -> *mut u8 {
        // An associated const is only evaluated where it is named, so naming it
        // here is what makes the alignment assert run for this `T`.
        let () = Self::_PAYLOAD_ALIGNED;
        let layout = Self::block_layout(len);
        // Zeroed rather than uninitialized: the header word in the prefix must
        // read as clear flags (see the module comment), and clearing the payload
        // too means a caller that reads before writing sees zeros rather than
        // whatever the allocator held.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        if base.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe {
            let block = base.add(Self::BASE_OFFSET);
            *(block.sub(BLOCK_PREFIX) as *mut u64) = layout.size() as u64;
            *(block.add(Self::LENGTH_OFFSET) as *mut usize) = len;
            block
        }
    }

    fn block_layout(len: usize) -> Layout {
        let items = std::mem::size_of::<T>()
            .checked_mul(len)
            .expect("virtualizable array payload size overflowed");
        let total = Self::BASE_OFFSET
            .checked_add(Self::ITEMS_OFFSET)
            .and_then(|prefix| prefix.checked_add(items))
            .expect("virtualizable array block size overflowed");
        let align = block_align(std::mem::align_of::<T>());
        Layout::from_size_align(total, align).expect("virtualizable array block layout")
    }

    fn items_ptr(&self) -> *mut T {
        unsafe { self.block.add(Self::ITEMS_OFFSET) as *mut T }
    }
}

impl<T: Copy> Drop for VirtArray<T> {
    fn drop(&mut self) {
        // The block pointer is not the allocation base — the size slot, the
        // header word and any alignment padding precede it — so the size slot
        // is what says how much to release, exactly as `free_off_gc_jitframe`
        // reads it, and the base is the block stepped back over the whole
        // prefix rather than over the size slot alone.
        unsafe {
            let base = self.block.sub(Self::BASE_OFFSET);
            let total = *(self.block.sub(BLOCK_PREFIX) as *const u64) as usize;
            let align = block_align(std::mem::align_of::<T>());
            let layout = Layout::from_size_align(total, align)
                .expect("virtualizable array block size slot was corrupted");
            std::alloc::dealloc(base, layout);
        }
    }
}

impl<T: Copy> std::ops::Deref for VirtArray<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.items_ptr(), self.len()) }
    }
}

impl<T: Copy> std::ops::DerefMut for VirtArray<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.items_ptr(), self.len()) }
    }
}

impl<T: Copy> Default for VirtArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> Clone for VirtArray<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for VirtArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Copy> From<&[T]> for VirtArray<T> {
    fn from(items: &[T]) -> Self {
        Self::from_slice(items)
    }
}

impl<T: Copy> FromIterator<T> for VirtArray<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        Self::from_slice(&items)
    }
}

impl<T: Copy> sealed::PointerShaped for VirtArray<T> {}

impl<T: Copy> VirtArrayBacking for VirtArray<T> {
    type Item = T;
    const LENGTH_OFFSET: usize = VirtArray::<T>::LENGTH_OFFSET;
    const ITEMS_OFFSET: usize = VirtArray::<T>::ITEMS_OFFSET;

    fn filled(value: T, len: usize) -> Self {
        VirtArray::filled(value, len)
    }
}

/// Register one `[..; virt]` state field on `info`, in the shape its declared
/// container has.
///
/// `witness` names the field's Rust type and is not called: a generated builder
/// runs before any state instance exists, so the backing has to be resolved from
/// the type rather than from a value.
///
/// The container's length and payload offsets register as an ordinary
/// array-pointer field, which is what the compiled entry reloads from.
pub fn register_virt_array_field<S, B: VirtArrayBacking>(
    info: &mut VirtualizableInfo,
    name: &str,
    item_type: Type,
    item_size: usize,
    field_offset: usize,
    witness: impl Fn(&S) -> &B,
) {
    let _ = witness;
    // `virtualizable.py cpu.arraydescrof` — the descr an element access is
    // emitted with. Its base is the payload offset, so `GETARRAYITEM_GC_*`
    // against the pointer the field holds addresses item `i` directly, with
    // no step in between for the entry to express.
    info.add_array_field(
        name,
        item_type,
        field_offset,
        B::LENGTH_OFFSET,
        B::ITEMS_OFFSET,
        majit_ir::descr::make_array_descr(B::ITEMS_OFFSET, item_size, item_type),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_is_one_pointer_word_with_its_length_and_payload_at_fixed_offsets() {
        let array = VirtArray::filled(7i64, 3);
        // What the state field holds, i.e. what `GETFIELD_GC_R` would load.
        let field: usize = unsafe { *(&array as *const VirtArray<i64> as *const usize) };
        assert_ne!(field, 0, "an empty-or-not block is never null");

        let length = unsafe { *((field + VirtArray::<i64>::LENGTH_OFFSET) as *const usize) };
        assert_eq!(length, 3);

        for index in 0..3 {
            let item =
                unsafe { *((field + VirtArray::<i64>::ITEMS_OFFSET + index * 8) as *const i64) };
            assert_eq!(item, 7, "item {index} read through the block offsets");
        }
    }

    #[test]
    fn the_header_word_in_front_of_a_block_reads_as_clear_flags() {
        let array = VirtArray::filled(0i64, 1);
        let field = unsafe { *(&array as *const VirtArray<i64> as *const usize) };
        // The word a header-relative probe would land on. In-bounds because the
        // prefix reserves it, and zero because the block is allocated zeroed.
        let header = unsafe { *((field - BLOCK_HEADER) as *const u64) };
        assert_eq!(header, 0);
    }

    #[test]
    fn resize_keeps_the_overlap_and_fills_the_rest() {
        let mut array = VirtArray::from_slice(&[1i64, 2, 3]);
        array.resize(5, 9);
        assert_eq!(&array[..], &[1, 2, 3, 9, 9]);
        array.resize(2, 0);
        assert_eq!(&array[..], &[1, 2]);
    }

    #[test]
    fn a_resize_to_the_same_length_keeps_the_block() {
        let mut array = VirtArray::from_slice(&[1i64, 2, 3]);
        let before = array.as_ptr();
        array.resize(3, 0);
        assert_eq!(
            array.as_ptr(),
            before,
            "an unchanged length reallocates nothing"
        );
        assert_eq!(&array[..], &[1, 2, 3]);
    }

    #[test]
    fn an_empty_array_still_has_a_block_to_address_item_zero_from() {
        let array: VirtArray<i64> = VirtArray::new();
        assert_eq!(array.len(), 0);
        assert!(array.is_empty());
        // The field's own word, not the slice's data pointer: the slice reports
        // a dangling non-null address for an empty length either way, and it is
        // the field the item-0 arithmetic starts from.
        let field: usize = unsafe { *(&array as *const VirtArray<i64> as *const usize) };
        assert_ne!(field, 0);
    }

    #[test]
    fn a_float_payload_is_eight_byte_aligned_wherever_the_word_size_lands() {
        let array = VirtArray::filled(1.5f64, 2);
        assert_eq!(array.as_ptr() as usize % 8, 0);
        assert_eq!(&array[..], &[1.5, 1.5]);
        assert!(VirtArray::<f64>::ITEMS_OFFSET >= std::mem::size_of::<usize>());
        assert_eq!(VirtArray::<f64>::ITEMS_OFFSET % 8, 0);
    }

    #[test]
    fn the_slice_surface_is_the_one_a_vec_presents() {
        let mut array = VirtArray::filled(0i64, 4);
        array.copy_from_slice(&[4, 3, 2, 1]);
        array[0] = 40;
        assert_eq!(array.len(), 4);
        assert_eq!(array.iter().copied().sum::<i64>(), 46);
        assert_eq!(array.to_vec(), vec![40, 3, 2, 1]);
        array.clear();
        assert_eq!(array.len(), 0);
    }

    #[test]
    fn a_virt_array_declares_its_block_offsets() {
        assert_eq!(
            <VirtArray<i64> as VirtArrayBacking>::LENGTH_OFFSET,
            VirtArray::<i64>::LENGTH_OFFSET,
        );
        assert_eq!(
            <VirtArray<i64> as VirtArrayBacking>::ITEMS_OFFSET,
            VirtArray::<i64>::ITEMS_OFFSET,
        );
    }

    /// A field holding the block pointer registers as an ordinary array-pointer
    /// field, which is the storage `compile.py:441-457` can rebuild from.
    #[test]
    fn a_block_backed_field_registers_as_an_array_pointer_field() {
        #[repr(C)]
        struct State {
            regs: VirtArray<i64>,
        }

        let mut info = VirtualizableInfo::without_vable_token();
        register_virt_array_field(
            &mut info,
            "regs",
            Type::Int,
            8,
            std::mem::offset_of!(State, regs),
            |s: &State| &s.regs,
        );

        let field = &info.array_fields[0];
        assert_eq!(
            field.storage,
            crate::virtualizable::VableArrayStorage::DirectPointer
        );
        assert_eq!(field.length_offset, VirtArray::<i64>::LENGTH_OFFSET);
        assert_eq!(field.items_offset, VirtArray::<i64>::ITEMS_OFFSET);
        assert!(field.can_read_length_from_heap());

        // The element descr addresses the payload from the pointer the field
        // holds, so an entry needs no step between the two loads.
        let descr = info.array_item_descr(0);
        let descr = descr.as_array_descr().expect("an array descr");
        assert_eq!(descr.base_size(), VirtArray::<i64>::ITEMS_OFFSET);
        assert_eq!(descr.item_size(), 8);
    }

    /// The heap-side readers the blackhole and the resume writer use reach the
    /// items of a block-backed field through its registered offsets.
    #[test]
    fn the_vable_item_readers_reach_a_block_through_its_registered_offsets() {
        #[repr(C)]
        struct State {
            regs: VirtArray<i64>,
        }

        let mut info = VirtualizableInfo::without_vable_token();
        register_virt_array_field(
            &mut info,
            "regs",
            Type::Int,
            8,
            std::mem::offset_of!(State, regs),
            |s: &State| &s.regs,
        );

        let mut state = State {
            regs: VirtArray::from_slice(&[10i64, 20, 30]),
        };
        let vable = &mut state as *mut State as *mut u8;
        let array = &info.array_fields[0];

        assert_eq!(
            unsafe { crate::virtualizable::bhimpl_arraylen_vable(vable, array) },
            3
        );
        for (index, expected) in [10i64, 20, 30].into_iter().enumerate() {
            assert_eq!(
                unsafe { crate::virtualizable::vable_read_array_item(vable, array, index as i64) },
                expected
            );
        }
        unsafe { crate::virtualizable::vable_write_array_item(vable, array, 1, 99) };
        assert_eq!(state.regs.to_vec(), vec![10, 99, 30]);
    }
}
