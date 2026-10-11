use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::pyobject::PyObject;
use crate::{PY_NULL, PyObjectRef};

/// Host constructor for a 3.14t length cell. Upstream `l.length`
/// (`rlist.py`) is a plain Signed. `AtomicUsize::new` is identity on
/// the inner usize (`std_identity`), so looking inside records the
/// field write rather than a residual.
pub(crate) fn length_cell(n: usize) -> AtomicUsize {
    AtomicUsize::new(n)
}

/// rlist.py `_ll_list_resize_hint_really` over-allocation with Signed
/// `newsize` (`@signature(..., types.int(), ...)`):
/// `some = 3 if newsize < 9 else 6; some += newsize >> 3; return newsize + some`.
pub(crate) fn rlist_overallocate_signed(newsize: i64) -> i64 {
    let some = if newsize < 9 { 3i64 } else { 6i64 };
    let some = some.wrapping_add(newsize >> 3);
    newsize.wrapping_add(some)
}

/// [`rlist_overallocate_signed`] for a `usize` capacity.
pub(crate) fn rlist_overallocate_cap(min_cap: usize) -> usize {
    rlist_overallocate_signed(min_cap as i64) as usize
}

/// `rstr.py` `AbstractStringRepr.ll_strcmp` body. No `stroruni.cmp`
/// oopspec: that hint belongs on a function whose first argument is
/// `Ptr(STR)` / `Ptr(UNICODE)`. A byte-slice helper is `Other` and
/// `_handle_stroruni_call` panics.
#[majit_macros::elidable]
pub(crate) fn ll_chars_strcmp(left: &[u8], right: &[u8]) -> isize {
    let cmplen = left.len().min(right.len());
    let left_p = left.as_ptr();
    let right_p = right.as_ptr();
    let mut i = 0usize;
    while i < cmplen {
        let diff = unsafe { *left_p.add(i) as isize - *right_p.add(i) as isize };
        if diff != 0 {
            return diff;
        }
        i += 1;
    }
    left.len() as isize - right.len() as isize
}

// The `GcArray(Float)` / `GcArray(Signed)` body and its no-collect allocation
// live with `rbigint`, whose `_digits` is lowered to one
// (`majit_rlib::lltypesystem::rlist`). The stable-tier constructors below are
// this crate's, because they root through this crate's shadow stack.
// The two `GcArray` tids are read through accessors rather than re-exported as
// constants: they are slots in *this* host's type registry, so pyre declares
// them from its own `gc.register_type` results (`pyre-jit/src/eval.rs`) instead
// of majit-rlib carrying pyre's numbering.
pub use majit_rlib::lltypesystem::rlist::{
    TYPED_ITEMS_BLOCK_ITEMS_OFFSET, TYPED_ITEMS_BLOCK_LEN_OFFSET, TypedItemsBlock,
    alloc_typed_items_block_immortal, alloc_typed_items_block_nursery, gc_float_array_gc_type_id,
    gc_int_array_gc_type_id, itemsblock_gc_enabled, set_gc_float_array_gc_type_id,
    set_gc_int_array_gc_type_id, try_alloc_typed_items_block_nursery, try_typed_items_block_layout,
    typed_items_block_capacity, typed_items_block_clear, typed_items_block_items_base,
    typed_items_block_layout,
};

/// GC type id for the variable-length backing block of
/// `W_ListObject.items` / `W_TupleObject.wrappeditems` /
/// `DictStorage.values`. Shape matches RPython's
/// `GcArray(OBJECTPTR)` from `rpython/rtyper/lltypesystem/rlist.py:84,116`
/// — a `T_IS_VARSIZE` block with a single-slot `capacity` header followed by
/// inline `PyObjectRef` items. Registered from [`ITEMS_BLOCK_TOKEN`] with
/// `items_have_gc_ptrs=true` so the GC walks each item slot as a Ref.
/// Re-exported from `pyre_jit_trace::descr` for existing call sites.
pub const PY_OBJECT_ARRAY_GC_TYPE_ID: u32 = 9;

// 41 is `Ptr(GcArray(Signed))` and 42 is `Ptr(GcArray(Float))`, re-exported
// above: registered after the dictview (39) and getset-property (40) slots, so
// `gc.register_type` returns those two here.

/// GC type id for the `W_ObjectObject.storage` block — the mapdict instance
/// attribute-value array (`mapdict.py _mapdict_init_empty` `self.storage`, a
/// `Ptr(GcArray(OBJECTPTR))`). Distinct from `PY_OBJECT_ARRAY_GC_TYPE_ID` (9,
/// list/tuple items) so the block keeps its own identity even though every slot
/// is now a reference: a boxed attribute's value, or an
/// `UnboxedPlainAttribute`'s longlong list as a varsize leaf GcArray. The tid
/// is registered `items_have_gc_ptrs=true`, so the collector walks the block's
/// interior itself and a barrier naming the block records a real edge; the
/// owning instance's `object_object_custom_trace` walks the same slots
/// (`instance_walk_boxed_storage`), which is idempotent.
/// Registered at the tail of the tid chain (after `W_COMPLEX_GC_TYPE_ID = 54`)
/// so no hardcoded constant shifts.
pub const W_MAPDICT_STORAGE_GC_TYPE_ID: u32 = 55;

/// The `(base_size, item_size, len_offset)` triple describing one varsize
/// GcArray block, derived from the Rust struct that carries the block's GC type
/// id — `symbolic.py get_array_token`.
///
/// Upstream computes all three numbers from a single array lltype and hands
/// that one triple to every consumer: the JIT's array descriptor
/// (`descr.py get_array_descr`), its length descriptor
/// (`descr.py get_field_arraylen_descr`) and the collector's varsize shape
/// (`gctypelayout.py encode_type_shape`, where `ofstovar`,
/// `varitemsize` and `ofstolength` all read the same `ARRAY`). Picking the
/// numbers apart per call site is what let the two scalar-array tids be
/// registered with `GcTypedArray`'s bare length word while the blocks actually
/// allocated under them were [`TypedItemsBlock`]s: the two structs agree on
/// 64-bit and differ by four bytes on wasm32, where the `[u64; 0]` items pad a
/// 4-byte length prefix out to 8. Every minor-collection copy of an
/// `rbigint._digits` block was then sized four bytes short and dropped the high
/// half of its last digit.
///
/// One token per (struct, element type) pair, so a consumer takes all three
/// numbers from the struct it means, or none of them.
pub struct ArrayToken {
    /// Offset of items[0] — the collector's `ofstovar`, the array descriptor's
    /// `base_size`.
    pub base_size: usize,
    /// `sizeof(ARRAY.OF)` — the element stride.
    pub item_size: usize,
    /// Offset of the length header — the collector's `ofstolength`.
    pub len_offset: usize,
}

/// `#[repr(C)] { capacity, items: [PyObjectRef; 0] }` — the single-block
/// inline-varsize GcArray body used by `W_ListObject.items` /
/// `W_TupleObject.items` / `DictStorage.values`.
/// Shape matches RPython's `GcArray(OBJECTPTR)` from
/// `rpython/rtyper/lltypesystem/rlist.py:84`: a length header at
/// offset 0 followed by inline items. Upstream's GcArray length IS
/// the allocated capacity (rlist.py:251 `len(l.items)` = allocated
/// slot count, fixed for the block's lifetime); live list length
/// lives on the enclosing `W_ListObject` wrapper per rlist.py:116
/// `("length", Signed)`.
///
/// Layout: offset 0 = `capacity` (= GcArray length header),
/// offset 8 = items[0..capacity]. Total allocation size =
/// `ITEMS_BLOCK_ITEMS_OFFSET + capacity * sizeof(PyObjectRef)`.
///
/// The header layout matches upstream and, by default, the allocator
/// now routes object-strategy blocks through the moving nursery
/// (`alloc_list_items_block_gc` / `alloc_tuple_items_block_gc` /
/// `grow_list_items_block_gc` → `try_gc_alloc(PY_OBJECT_ARRAY_GC_TYPE_ID,
/// cap)`), so `PY_OBJECT_ARRAY_GC_TYPE_ID` and the list/tuple custom
/// traces that forward the block-pointer field (eval.rs
/// `list_object_custom_trace` / `tuple_object_custom_trace`) are live at
/// collection time. `W_ListObject` / `W_TupleObject` hold
/// `{length: usize, items: *mut ItemsBlock}` fields directly (no
/// `PyObjectArray` fat wrapper for list/tuple). The `std::alloc`
/// `alloc_items_block` / `grow_items_block` below are the
/// `MAJIT_GC_ITEMSBLOCK=0` fallback (provably identical pre-migration
/// behaviour).
#[repr(C)]
#[majit_macros::jit_immutable_fields("capacity")]
pub struct ItemsBlock {
    /// Allocated capacity — treated as the GcArray-length header
    /// (rlist.py:251 `len(l.items)`). The GC registration sets
    /// `length_offset=0` to this field so the walker iterates
    /// `0..capacity`. Fixed from `alloc_items_block()` through
    /// `dealloc_items_block()`; a `grow_items_block()` call
    /// allocates a fresh block rather than mutating this field.
    pub capacity: usize,
    /// Items inline after the header. Size known only at allocation
    /// time — accessed via pointer arithmetic from
    /// `items_block_items_base()`.
    items: [PyObjectRef; 0],
}

pub const ITEMS_BLOCK_ITEMS_OFFSET: usize = std::mem::offset_of!(ItemsBlock, items);

/// Offset of the `capacity` header the collector reads as the GcArray length.
pub const ITEMS_BLOCK_LEN_OFFSET: usize = std::mem::offset_of!(ItemsBlock, capacity);

/// `get_array_token(GcArray(OBJECTPTR))` — the shape [`PY_OBJECT_ARRAY_GC_TYPE_ID`]
/// is registered with and `pyobject_gcarray_descr` carries.
pub const ITEMS_BLOCK_TOKEN: ArrayToken = ArrayToken {
    base_size: ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: std::mem::size_of::<PyObjectRef>(),
    len_offset: ITEMS_BLOCK_LEN_OFFSET,
};

/// Return the items base pointer (i.e. `&items[0]`) of an
/// `ItemsBlock`. Null-safe: returns a null `*mut PyObjectRef` if the
/// block itself is null, so callers can treat a null items pointer as
/// an empty list without branching through `Option`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn items_block_items_base(block: *mut ItemsBlock) -> *mut PyObjectRef {
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET) as *mut PyObjectRef }
}

/// `setarrayitem_gc` for a type-9 `ItemsBlock`:
/// `MiniMarkGC.write_barrier_from_array(addr_array, index)` then the store.
/// Card marking records the written page; an array without `HAS_CARDS` falls
/// back to `remember_young_pointer`. A raw write after a barrier that ran
/// in another function leaves a window in which a collection can consume
/// the remembered-set entry and reset the flag before the young pointer
/// lands.
///
/// # Safety
/// `block` is a live `ItemsBlock` and `index` is in range.
#[inline]
pub unsafe fn items_block_set_ref(block: *mut ItemsBlock, index: usize, value: PyObjectRef) {
    debug_assert!(!block.is_null(), "setarrayitem_gc on a null ItemsBlock");
    let header = unsafe { majit_gc::header::header_of(block as usize) };
    if unsafe { (*header).has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS) }
        && unsafe { (*header).is_forwarded() }
    {
        stale_array_abort(block as usize, index);
    }
    majit_gc::gc_write_barrier_from_array(majit_ir::GcRef(block as usize), index);
    unsafe { *items_block_items_base(block).add(index) = value };
}

unsafe fn arraycopy_store_item(
    block: *mut ItemsBlock,
    index: usize,
    value: PyObjectRef,
    dest_managed: bool,
) {
    if dest_managed {
        unsafe { items_block_set_ref(block, index, value) };
    } else {
        unsafe { *items_block_items_base(block).add(index) = value };
    }
}

/// `rgc.ll_arraymove(array, source_start, dest_start, length)` — runtime
/// target for the `list.ll_arraymove` oopspec (OS_ARRAYMOVE / 9).
///
/// RPython's native path calls `gc_writebarrier_before_move` when the array
/// item contains GC pointers, casts the two item addresses to raw addresses,
/// and invokes `raw_memmove_no_free`. The residual ABI deliberately keeps
/// only those same four arguments. For a managed block, the existing GC
/// TYPE_INFO row supplies the array token and `_contains_gcptr(TYPE.OF)` bit;
/// a raw fallback block uses the `GcArray(PyObjectRef)` token this adapter
/// boundary allocates. `ptr::copy` has memmove's overlap contract.
pub extern "C" fn jit_ll_arraymove(
    array: crate::PyObjectRef,
    source_start: i64,
    dest_start: i64,
    length: i64,
) {
    if array.is_null() || length <= 0 {
        return;
    }
    assert!(
        source_start >= 0,
        "ll_arraymove source_start must be non-negative"
    );
    assert!(
        dest_start >= 0,
        "ll_arraymove dest_start must be non-negative"
    );

    // A residual call copies its argument out of the jitframe before it
    // enters host Rust.  A collection which ran at that boundary rewrites the
    // jitframe/root slot but not this ABI copy, so follow the forwarding word
    // before the layout query, the barriers and the copy all read the header.
    // `gc_shrink_array` resolves the same way for the same reason; RPython's
    // GC transform makes the equivalent reload around `rgc.ll_arraymove`
    // automatically.
    let address =
        crate::gc_hook::try_gc_current_object_address(array as crate::gc_hook::GCREF) as usize;
    let layout = majit_gc::gc_varsize_layout(address).unwrap_or(majit_gc::GcVarSizeLayout {
        base_size: ITEMS_BLOCK_TOKEN.base_size,
        item_size: ITEMS_BLOCK_TOKEN.item_size,
        // The admitted unmanaged fallback is GcArray(PyObjectRef). Asking its
        // barrier hook is harmless when no GC owns the pointer.
        items_have_gc_ptrs: true,
    });
    let source_offset = (source_start as usize)
        .checked_mul(layout.item_size)
        .and_then(|offset| layout.base_size.checked_add(offset))
        .expect("ll_arraymove source address overflow");
    let dest_offset = (dest_start as usize)
        .checked_mul(layout.item_size)
        .and_then(|offset| layout.base_size.checked_add(offset))
        .expect("ll_arraymove destination address overflow");
    let byte_length = (length as usize)
        .checked_mul(layout.item_size)
        .expect("ll_arraymove byte length overflow");

    if layout.items_have_gc_ptrs {
        if length == 1 {
            // rgc.py `ll_arraymove`: the literal `copy_item` head exists for
            // EffectInfo and performs an ordinary GC-pointer store.
            crate::gc_hook::try_gc_write_barrier(address as crate::gc_hook::GCREF);
        } else {
            crate::gc_hook::try_gc_write_barrier_before_move(address as crate::gc_hook::GCREF);
        }
    }
    unsafe {
        std::ptr::copy(
            (address as *const u8).add(source_offset),
            (address as *mut u8).add(dest_offset),
            byte_length,
        );
    }
}

/// `rgc.ll_arraycopy(source, dest, source_start, dest_start, length)` —
/// runtime target for the `list.ll_arraycopy` oopspec (OS_ARRAYCOPY / 1).
///
/// rgc.py `ll_arraycopy`: `copy_item` when `length <= 1`; otherwise
/// `writebarrier_before_copy` when the item contains GC pointers, then
/// `raw_memcopy`. `false` from the barrier is the per-item `copy_item`
/// slow path. `rgc.py ll_arraycopy`'s `@jit.oopspec`: a call from an
/// interpreter body is the `OS_ARRAYCOPY` residual, never a look-inside.
/// `dont_look_inside_cannot_raise` matches `_canraise` False on that
/// helper (`call.py`): the `assert!` on negative starts is a native
/// contract, not a Python exception. Without it the residual is
/// `CallMayForceR` and a transparent BINARY_SLICE walk declines.
#[majit_macros::dont_look_inside_cannot_raise]
#[majit_macros::oopspec("list.ll_arraycopy(source, dest, source_start, dest_start, length)")]
pub extern "C" fn jit_ll_arraycopy(
    source: crate::PyObjectRef,
    dest: crate::PyObjectRef,
    source_start: i64,
    dest_start: i64,
    length: i64,
) {
    if source.is_null() || dest.is_null() || length <= 0 {
        return;
    }
    assert!(
        source_start >= 0,
        "ll_arraycopy source_start must be non-negative"
    );
    assert!(
        dest_start >= 0,
        "ll_arraycopy dest_start must be non-negative"
    );

    // Publish both blocks before any GC hook. The second
    // `try_gc_current_object_address` and the dest write-barrier can wait
    // behind a collection; that collection rewrites shadow-stack slots,
    // not native copies of the ABI arguments.
    let _roots = crate::gc_roots::push_roots();
    let source_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(source as crate::pyobject::PyObjectRef);
    let dest_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dest as crate::pyobject::PyObjectRef);

    let follow = |slot: usize| {
        crate::gc_hook::try_gc_current_object_address(
            crate::gc_roots::shadow_stack_get(slot) as crate::gc_hook::GCREF
        ) as usize
    };
    let dest_address = follow(dest_slot);
    let source_address = follow(source_slot);
    let fallback = majit_gc::GcVarSizeLayout {
        base_size: ITEMS_BLOCK_TOKEN.base_size,
        item_size: ITEMS_BLOCK_TOKEN.item_size,
        items_have_gc_ptrs: true,
    };
    // An unmanaged typed block has no registry entry. If only one
    // endpoint is registered, reuse that layout for both; the fallback
    // is only for two unknown addresses. Mixed ownership still copies
    // with that layout, but the GC barrier/setter only run on a
    // registered dest (and the 5-arg copy barrier needs both headers).
    let source_layout_opt = majit_gc::gc_varsize_layout(source_address);
    let dest_layout_opt = majit_gc::gc_varsize_layout(dest_address);
    let dest_managed = dest_layout_opt.is_some();
    let source_managed = source_layout_opt.is_some();
    let (source_layout, dest_layout) = match (source_layout_opt, dest_layout_opt) {
        (Some(source), Some(dest)) => (source, dest),
        (Some(known), None) | (None, Some(known)) => (known, known),
        (None, None) => (fallback, fallback),
    };
    // rgc.py `ll_arraycopy`: `assert TP == typeOf(dest).TO`.
    assert!(
        source_layout.item_size == dest_layout.item_size
            && source_layout.items_have_gc_ptrs == dest_layout.items_have_gc_ptrs,
        "ll_arraycopy source and dest must have the same item layout"
    );
    let source_offset = (source_start as usize)
        .checked_mul(source_layout.item_size)
        .and_then(|offset| source_layout.base_size.checked_add(offset))
        .expect("ll_arraycopy source address overflow");
    let dest_offset = (dest_start as usize)
        .checked_mul(dest_layout.item_size)
        .and_then(|offset| dest_layout.base_size.checked_add(offset))
        .expect("ll_arraycopy destination address overflow");
    let byte_length = (length as usize)
        .checked_mul(dest_layout.item_size)
        .expect("ll_arraycopy byte length overflow");

    // rgc.py `ll_arraycopy`: length <= 1 is `copy_item` (a GC-pointer
    // store), not the bulk barrier-plus-memcpy body. Unmanaged dests
    // have no GC header; `items_block_set_ref` would `header_of` them.
    if dest_layout.items_have_gc_ptrs && length == 1 {
        let dest_address = follow(dest_slot);
        let source_address = follow(source_slot);
        let value = unsafe {
            *((source_address as *const u8).add(source_offset) as *const crate::PyObjectRef)
        };
        unsafe {
            arraycopy_store_item(
                dest_address as *mut ItemsBlock,
                dest_start as usize,
                value,
                dest_managed,
            );
        }
        return;
    }

    let mut slowpath = false;
    if dest_layout.items_have_gc_ptrs && dest_managed {
        if source_managed {
            slowpath = !majit_gc::gc_writebarrier_before_copy(
                majit_ir::GcRef(follow(source_slot)),
                majit_ir::GcRef(follow(dest_slot)),
                source_start as usize,
                dest_start as usize,
                length as usize,
            );
        } else {
            slowpath = true;
        }
    }
    if slowpath {
        let mut i = 0i64;
        while i < length {
            let dest_address = follow(dest_slot);
            let source_address = follow(source_slot);
            let value = unsafe {
                *((source_address as *const u8)
                    .add(source_offset + (i as usize) * source_layout.item_size)
                    as *const crate::PyObjectRef)
            };
            unsafe {
                arraycopy_store_item(
                    dest_address as *mut ItemsBlock,
                    dest_start as usize + i as usize,
                    value,
                    dest_managed,
                );
            }
            i += 1;
        }
        return;
    }
    let dest_address = follow(dest_slot);
    let source_address = follow(source_slot);
    unsafe {
        std::ptr::copy_nonoverlapping(
            (source_address as *const u8).add(source_offset),
            (dest_address as *mut u8).add(dest_offset),
            byte_length,
        );
    }
}

/// rgc.py `ll_arraycopy` for the object `ItemsBlock` ARRAY.
///
/// `@specialize.ll()` gives every ARRAY its own `ll_arraycopy` graph, and its
/// `length <= 1` `copy_item` head ("Hack to ensure that we get a proper
/// effectinfo.write_descrs_arrays") is the getarrayitem / setarrayitem
/// writeanalyze reads that ARRAY's effects off. Longer copies run the shared
/// barrier-plus-memcpy body, [`jit_ll_arraycopy`].
///
/// # Safety
/// `source` and `dest` are live `ItemsBlock`s and both ranges are in bounds.
#[majit_macros::oopspec("list.ll_arraycopy(source, dest, source_start, dest_start, length)")]
pub unsafe fn ll_arraycopy_items_block(
    source: *mut ItemsBlock,
    dest: *mut ItemsBlock,
    source_start: i64,
    dest_start: i64,
    length: i64,
) {
    if length <= 1 {
        if length == 1 {
            // rgc.py `copy_item`: `dest[dest_start] = source[source_start]`.
            let item = unsafe { *items_block_items_base(source).add(source_start as usize) };
            let dest_managed = majit_gc::gc_varsize_layout(dest as usize).is_some();
            unsafe { arraycopy_store_item(dest, dest_start as usize, item, dest_managed) };
        }
        return;
    }
    jit_ll_arraycopy(
        source as PyObjectRef,
        dest as PyObjectRef,
        source_start,
        dest_start,
        length,
    );
}

/// [`ll_arraycopy_items_block`] for the `GcArray(Signed)` ARRAY.
///
/// # Safety
/// `source` and `dest` are live int `TypedItemsBlock`s and both ranges are in
/// bounds.
#[majit_macros::oopspec("list.ll_arraycopy(source, dest, source_start, dest_start, length)")]
pub unsafe fn ll_arraycopy_int_items_block(
    source: *mut TypedItemsBlock,
    dest: *mut TypedItemsBlock,
    source_start: i64,
    dest_start: i64,
    length: i64,
) {
    if length <= 1 {
        if length == 1 {
            // rgc.py `copy_item`: `dest[dest_start] = source[source_start]`.
            let item = unsafe {
                *(typed_items_block_items_base(source) as *const i64).add(source_start as usize)
            };
            unsafe {
                *(typed_items_block_items_base(dest) as *mut i64).add(dest_start as usize) = item
            };
        }
        return;
    }
    jit_ll_arraycopy(
        source as PyObjectRef,
        dest as PyObjectRef,
        source_start,
        dest_start,
        length,
    );
}

/// [`ll_arraycopy_items_block`] for the `GcArray(Float)` ARRAY.
///
/// `source` and `dest` are [`FloatItemsBlock`] pointers: `do_fixed_list_ll_arraycopy`
/// names the copy's ARRAY from `dest`'s type, and a `TypedItemsBlock` there
/// names `GcArray(Signed)`.
///
/// # Safety
/// `source` and `dest` are live float items blocks and both ranges are in
/// bounds.
#[majit_macros::oopspec("list.ll_arraycopy(source, dest, source_start, dest_start, length)")]
pub unsafe fn ll_arraycopy_float_items_block(
    source: *mut FloatItemsBlock,
    dest: *mut FloatItemsBlock,
    source_start: i64,
    dest_start: i64,
    length: i64,
) {
    if length <= 1 {
        if length == 1 {
            // rgc.py `copy_item`: `dest[dest_start] = source[source_start]`.
            let item = unsafe {
                *(typed_items_block_items_base(source as *mut TypedItemsBlock) as *const f64)
                    .add(source_start as usize)
            };
            unsafe {
                *(typed_items_block_items_base(dest as *mut TypedItemsBlock) as *mut f64)
                    .add(dest_start as usize) = item
            };
        }
        return;
    }
    jit_ll_arraycopy(
        source as PyObjectRef,
        dest as PyObjectRef,
        source_start,
        dest_start,
        length,
    );
}

/// Allocated capacity (GcArray length header) of an `ItemsBlock`.
/// Returns 0 for a null pointer so "empty list" is represented by
/// a null `items` field.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn items_block_capacity(block: *mut ItemsBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

/// Allocate a fresh `ItemsBlock` populated with the given values. The
/// capacity is `values.len().max(1)`; unused slots past `values.len()`
/// are NULL-initialised so the GC walker (once Phase L2 activates
/// `PY_OBJECT_ARRAY_GC_TYPE_ID`) sees valid NULL refs past the live
/// prefix — upstream `gc_malloc_array` zero-fills (rlist.py:262-267
/// `_ll_list_resize_really`). Used by `W_ListObject::from_vec`.
///
/// The `max(1)` clamp is the list-strategy overallocation policy
/// (rlist.py:251 `_ll_list_resize_*` always keeps at least one slot
/// for in-place growth). Tuples must NOT use this allocator — see
/// [`alloc_tuple_items_block`] for the exact-size variant.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_list_items_block(values: &[PyObjectRef]) -> *mut ItemsBlock {
    let len = values.len();
    let cap = len.max(1);
    unsafe {
        let block = alloc_items_block(cap);
        let base = items_block_items_base(block);
        // `rlist.py` `_ll_list_resize` copies with `ll_setitem_fast` /
        // `ll_getitem_fast`, not `Enumerate`.
        let src = values.as_ptr();
        let mut i = 0usize;
        while i < len {
            *base.add(i) = *src.add(i);
            i += 1;
        }
        while i < cap {
            *base.add(i) = PY_NULL;
            i += 1;
        }
        block
    }
}

/// `pypy/objspace/std/tupleobject.py` `W_TupleObject`
/// allocator. Allocates an `ItemsBlock` with capacity exactly equal to
/// `values.len()` — tuples are immutable so the GcArray header
/// `length` IS the live tuple length (no overallocation room). For an
/// empty tuple this yields a 0-cap header-only block; the GcArray
/// pointer is non-null but addresses zero items.
///
/// Read length back via `arraylen_gc(items_block, pyobject_gcarray_descr)`
/// or [`items_block_capacity`] on the host side. No companion length
/// cache lives on `W_TupleObject` (`_immutable_fields_ =
/// ['wrappeditems[*]']` per upstream tupleobject.py:381).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_tuple_items_block(values: &[PyObjectRef]) -> *mut ItemsBlock {
    let cap = values.len();
    unsafe {
        let block = alloc_items_block(cap);
        let base = items_block_items_base(block);
        let src = values.as_ptr();
        let mut i = 0usize;
        while i < cap {
            *base.add(i) = *src.add(i);
            i += 1;
        }
        block
    }
}

/// Grow an `ItemsBlock` to `new_cap` capacity, copying `live_len`
/// existing items from `old` and NULL-initialising the rest. Returns the new
/// block. The caller swaps the owner field and then deallocates `old`, keeping
/// the SETFIELD_GC barrier adjacent to the publication. `old` may be null
/// (fresh allocation). rlist.py:262-267 parity.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn grow_list_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    unsafe { grow_items_block(old, new_cap, live_len) }
}

/// Deallocate an `ItemsBlock` previously allocated via
/// `alloc_list_items_block` / `grow_list_items_block`. No-op on null.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn dealloc_list_items_block(block: *mut ItemsBlock) {
    unsafe { dealloc_items_block(block) }
}

// ─── mapdict instance-storage block: nursery GcArray(OBJECTPTR) ───────────
//
// `W_ObjectObject.storage` (`mapdict.py _mapdict_init_empty` `self.storage`) is a
// `Ptr(GcArray(OBJECTPTR))`. It carries the same inline-traced shape as
// list/tuple item blocks (`PY_OBJECT_ARRAY_GC_TYPE_ID`) — every slot is a
// reference — under its own tid (`W_MAPDICT_STORAGE_GC_TYPE_ID`) and differs
// only in being allocated on the nursery bump (`malloc_varsize`). The
// instance is already a nursery object; `instance_walk_boxed_storage`
// rewrites its `storage` slot. A non-moving block was a shortcut so the
// instance's `storage` pointer never needed rewriting. That is not
// `mapdict.py`. The nursery bump is `malloc_varsize`.

/// Allocate a fresh nursery `ItemsBlock` holding `values` in its first slots and
/// NULL in the rest, tagged `W_MAPDICT_STORAGE_GC_TYPE_ID` (leaf). The map is
/// the length authority (mapdict.py), so `cap` is an allocation bound
/// rather than a length; a live instance passes the larger of its current
/// capacity and `values.len()` so the capacity never shrinks. Every slot is
/// written, so the block is safe to expose to the collector immediately. Falls
/// back to the `std::alloc` [`alloc_items_block`] when no GC hook is installed
/// (pure interpreter / early startup). `cap` 0 yields a header-only block.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_instance_items_block(values: &[PyObjectRef], cap: usize) -> *mut ItemsBlock {
    debug_assert!(cap >= values.len());
    let _roots = crate::gc_roots::push_roots();
    let values_base = crate::gc_roots::pin_roots(values);
    unsafe {
        let block = alloc_mapdict_storage_block(cap);
        let _ = crate::gc_roots::pin_root(block as PyObjectRef);
        let block =
            crate::gc_roots::shadow_stack_get(values_base + values.len()) as *mut ItemsBlock;
        let base = items_block_items_base(block);
        for i in 0..values.len() {
            *base.add(i) = crate::gc_roots::shadow_stack_get(values_base + i);
        }
        for i in values.len()..cap {
            *base.add(i) = PY_NULL;
        }
        block
    }
}

/// Grow (or shrink) an instance storage block to `new_cap`, copying `live_len`
/// existing items from `old` and NULL-filling any spare tail. The caller keeps
/// the owning instance rooted, installs and barriers the new block, then
/// deallocates `old`; dropping it here would introduce a GC operation while the
/// fresh block is not yet reachable from its owner. `old` may be null.
/// mapdict.py _set_mapdict_increase_storage1 `_add_attr` grow-by-one.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn grow_instance_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    unsafe {
        let _roots = crate::gc_roots::push_roots();
        // A null `old` is pinned too: the slot holds the null word, which no
        // collection rewrites, and every path then leaves the bracket at one
        // depth.
        let old_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(old as PyObjectRef);
        let fresh = alloc_mapdict_storage_block(new_cap);
        let old = crate::gc_roots::shadow_stack_get(old_slot) as *mut ItemsBlock;
        let new_base = items_block_items_base(fresh);
        let copy = live_len.min(new_cap);
        if !old.is_null() && copy > 0 {
            std::ptr::copy_nonoverlapping(items_block_items_base(old), new_base, copy);
        }
        for i in copy..new_cap {
            *new_base.add(i) = PY_NULL;
        }
        // Copied slots can be young. A nursery-full `fresh` block is
        // old-gen with TRACK_YOUNG_PTRS still set.
        crate::gc_hook::try_gc_write_barrier_managed(fresh as crate::gc_hook::GCREF);
        fresh
    }
}

/// Deallocate an instance storage block. A GC-managed (stable) block is
/// reclaimed by the collector and must never be freed here; the `std::alloc`
/// fallback block is freed. No-op on null. `try_gc_owns_object` discriminates.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn dealloc_instance_items_block(block: *mut ItemsBlock) {
    unsafe { dealloc_items_block(block) }
}

/// Nursery leaf-block allocator for mapdict storage (`malloc_varsize`).
/// `instance_walk_boxed_storage` forwards the instance's `storage` slot
/// and walks the items. Capacity is set; items are left uninitialised
/// (the caller writes every slot before exposing the block). Falls back
/// to `std::alloc` [`alloc_items_block`] when no GC hook is installed.
/// `cap` may be zero (header-only block).
unsafe fn alloc_mapdict_storage_block(cap: usize) -> *mut ItemsBlock {
    let payload = ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<PyObjectRef>();
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_MAPDICT_STORAGE_GC_TYPE_ID, payload);
    if !raw.is_null() {
        let block = raw as *mut ItemsBlock;
        unsafe { (*block).capacity = cap };
        return block;
    }
    unsafe { alloc_items_block(cap) }
}

/// Phase L2 nursery allocation of a fresh `ItemsBlock` with `cap` slots,
/// as a `PY_OBJECT_ARRAY_GC_TYPE_ID` varsize GcArray. The collector reads
/// its length from the offset-0 `capacity` header and forwards
/// items[0..capacity], so the caller MUST write every slot (value or
/// NULL) before a collection can observe the block. Falls back to the
/// `std::alloc` [`alloc_items_block`] when the gate is off or no GC hook
/// is installed (pure interpreter / early startup). Items are left
/// uninitialised either way. The bool is whether the block is GC-owned.
///
/// # Why `try_gc_alloc` and not the collecting hook
///
/// `try_gc_alloc` is the *no-collect* hook, so a block over the collector's
/// large-object threshold is born in the old generation and only a major
/// reclaims it. `malloc_varsize`'s large arm instead takes
/// `external_malloc(..., alloc_young=True)`, and every caller here already
/// brackets this call with `push_roots` and republishes the result through the
/// shadow stack, so the rooting discipline the collecting hook wants is
/// already met at these three sites.
///
/// What is not met is the invariant the rest of the interpreter rests on. The
/// house rule for a *stable* kind is one lifetime-scoped pin at the mint with
/// no read-back at later uses, and it is sound only because an old-generation
/// object cannot change address. A young rawmalloced object also cannot change
/// address — but it *dies* at the next minor if nothing reached it, so moving
/// this allocation to the collecting hook would make every unrooted holder
/// elsewhere wrong for a reason no relocation audit looks for. That is the same
/// blocker the born-old stepping stone in `gc_interp.rs` names, not a separate
/// one, and it is why the entry point here stays the no-collect one until the
/// root-set census is re-graded for liveness rather than for movement.
unsafe fn alloc_items_block_gc(cap: usize) -> (*mut ItemsBlock, bool) {
    unsafe { try_alloc_items_block_gc(cap).unwrap_or_else(|| items_block_alloc_failed(cap)) }
}

/// [`alloc_items_block_gc`] for a capacity that came from Python.
/// The bool is whether the block came from `try_gc_alloc` (GC-owned)
/// rather than `std::alloc`.
unsafe fn try_alloc_items_block_gc(cap: usize) -> Option<(*mut ItemsBlock, bool)> {
    if itemsblock_gc_enabled() {
        let payload = try_items_block_layout(cap)?.size();
        if let Some(raw) = crate::gc_hook::try_gc_alloc(PY_OBJECT_ARRAY_GC_TYPE_ID, payload)
            && !raw.is_null()
        {
            let block = raw as *mut ItemsBlock;
            unsafe {
                // IncrementalMiniMark does not zero-fill. Publish
                // `capacity` only after the payload is NULL: the type-9
                // walker traces `items[0..capacity]` as soon as this
                // block is a root, and callers `pin_root` it before
                // `try_gc_owns_object`. Zeroing the whole payload first
                // leaves a 0 length header so a concurrent walk cannot
                // follow dirty nursery words
                // (`rlist.py` `_ll_list_resize_hint_really` mallocs
                // through `gc_malloc_array`, which leaves GCREF slots
                // NULL).
                std::ptr::write_bytes(raw, 0, payload);
                (*block).capacity = cap;
            }
            return Some((block, true));
        }
    }
    unsafe { try_alloc_items_block(cap).map(|block| (block, false)) }
}

/// List-construction allocator on the Phase L2 nursery path. Pins each
/// element across the (collecting) block allocation and fills from the
/// relocated shadow-stack slots — the `pop_roots` read-back of
/// `w_tuple_new_array_backed` — because the local `values` slice still
/// holds pre-collection addresses. Capacity is `len.max(1)`
/// (overallocation policy); spare slots are NULL. Degrades to the
/// `std::alloc` [`alloc_list_items_block`] when the gate is off.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_list_items_block_gc(values: &[PyObjectRef]) -> *mut ItemsBlock {
    if !itemsblock_gc_enabled() {
        return unsafe { alloc_list_items_block(values) };
    }
    let len = values.len();
    let cap = len.max(1);
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::pin_roots(values);
    let block_slot = crate::gc_roots::shadow_stack_len();
    let (block, owns_block) = unsafe { alloc_items_block_gc(cap) };
    // `alloc_items_block_gc` returns a fresh GCREF.  RPython's
    // gct_fv_gc_malloc pop_roots makes that result live before the next
    // collecting operation, so the fresh array is published here rather than
    // carried only in this frame's native copy.
    let _ = crate::gc_roots::pin_root(block as PyObjectRef);
    // The barrier is the last thing before the items land, the way
    // `writebarrier_before_copy` precedes `ll_arraycopy`'s memcpy.  Running
    // the fill first would leave the young elements in an unregistered
    // old-gen block, where a minor collection does not trace them.
    // `owns_block` is the placement `try_alloc_items_block_gc` already
    // knew; the address is taken from the published slot, which is the one
    // authority a promotion rewrites.
    let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
    let base = unsafe { items_block_items_base(block) };
    // Old→young barrier if the block landed in old-gen (see
    // alloc_tuple_items_block_gc): registers an old-gen block holding young
    // elements onto the remembered set, no-op for a nursery block.
    if owns_block {
        crate::gc_hook::try_gc_write_barrier(block as crate::gc_hook::GCREF);
    }
    if len > 0 {
        // RPython's pop_roots reloads the livevars as one generated block.
        // Enter TLS once for the whole contiguous item range instead of once
        // per element; `base` names the freshly allocated block's initialized
        // prefix and cannot alias the shadow stack.
        let dst = unsafe { std::slice::from_raw_parts_mut(base, len) };
        crate::gc_roots::shadow_stack_copy_range(save, dst);
    }
    for i in len..cap {
        unsafe { *base.add(i) = PY_NULL };
    }
    block
}

/// List-grow allocator on the Phase L2 nursery path. Pins the OLD BLOCK across
/// the new (collecting) allocation, copies its `live_len` items into the new
/// one and NULL-fills the spare tail. The caller installs the new owner edge
/// through its write barrier and then hands the old block to
/// [`dealloc_list_items_block`] (which no-ops on a GC-managed block — the
/// collector reclaims it). `old` may be null with `live_len == 0` (fresh
/// allocation). Degrades to the `std::alloc` [`grow_list_items_block`] when the
/// gate is off.
///
/// The block itself is the root, not its items. A block is a traced GC array of
/// GC pointers (`PY_OBJECT_ARRAY_GC_TYPE_ID` is registered `TypeInfo::varsize`
/// over `PyObjectRef` slots, `pyre-jit/src/eval.rs`), so a collection during the
/// allocation below keeps every item reachable through it and relocates the
/// slots in place — reading them back afterwards yields post-collection
/// addresses without naming them one at a time. That is also the upstream
/// shape: `_ll_list_resize_really` (rlist.py) mallocs the new array and
/// `ll_arraycopy`s into it, with no per-item root bracket anywhere.
///
/// Pinning per item instead made a grow cost `2 * live_len` GC-ownership
/// queries — each a TLS lookup, a `RefCell` borrow, an arena range test and a
/// rawmalloced-set probe — so the 1% of appends that resize dominated the
/// profile of every Object-strategy list build.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn grow_list_items_block_gc(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    // rlist.py `_ll_list_resize_really`:
    //   newitems = malloc(LIST.items.TO, new_allocated)
    //   rgc.ll_arraycopy(items, newitems, 0, 0, p)
    // `items` is a livevar across the malloc.
    let roots = crate::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(old as PyObjectRef);
    let newitems = unsafe { ll_new_items_array(new_cap as i64) };
    if live_len > 0 {
        let _ = roots.pin_root(newitems as PyObjectRef);
        unsafe {
            ll_arraycopy_items_block(
                roots.get(base) as *mut ItemsBlock,
                roots.get(base + 1) as *mut ItemsBlock,
                0,
                0,
                live_len as i64,
            );
        }
        return roots.get(base + 1) as *mut ItemsBlock;
    }
    newitems
}

/// `malloc(LIST.items.TO, count)` for the object `items` ARRAY: a zeroed
/// `GcArray(OBJECTPTR)`.  A traced call is `new_array_clear`
/// (`jtransform.py do_fixed_newlist`, as for `_ll_alloc_and_clear`); the body
/// is the interpreter's allocation.
///
/// # Safety
/// `count` is a non-negative capacity.
#[inline(never)]
#[majit_macros::oopspec("newlist_clear(count)")]
unsafe fn ll_new_items_array(count: i64) -> *mut ItemsBlock {
    let count = count as usize;
    unsafe {
        let (block, _) = alloc_items_block_gc(count);
        let base = items_block_items_base(block);
        for i in 0..count {
            *base.add(i) = PY_NULL;
        }
        block
    }
}

/// [`grow_list_items_block_gc`] for a capacity that came from Python.  The
/// fresh block is the first step and the only fallible one, so a refusal
/// returns before any barrier runs and leaves the list on its old block.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn try_grow_list_items_block_gc(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> Option<*mut ItemsBlock> {
    if !itemsblock_gc_enabled() {
        return unsafe { try_grow_items_block(old, new_cap, live_len) };
    }
    let _roots = crate::gc_roots::push_roots();
    let old_slot = {
        // Publish the old block before *any* GC hook.  In particular,
        // `try_gc_owns_object` is not a harmless predicate: its cross-thread
        // path can wait behind a collection.  Asking whether `old` was
        // managed before rooting it therefore let that collection relocate
        // the block, after which the copy below read its reclaimed nursery
        // slots and installed stale item pointers in the new block.
        //
        // The shadow-stack walker ignores non-GC addresses, so rooting the
        // std::alloc fallback unconditionally is both safe and removes the
        // ownership-query safepoint entirely.  A null `old` is pinned the
        // same way, so every path publishes the same slots.
        let slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(old as crate::pyobject::PyObjectRef);
        slot
    };
    let new_block_slot = crate::gc_roots::shadow_stack_len();
    let (new_block, owns_new) = unsafe { try_alloc_items_block_gc(new_cap)? };
    let _ = crate::gc_roots::pin_root(new_block as PyObjectRef);
    // Barrier before the copy — see `alloc_list_items_block_gc` for why
    // the barrier cannot follow the items. `owns_new` is the placement
    // `try_alloc_items_block_gc` already knew.
    let new_block = crate::gc_roots::shadow_stack_get(new_block_slot) as *mut ItemsBlock;
    // Old→young barrier if the grown block landed in old-gen (see
    // alloc_tuple_items_block_gc).
    if owns_new {
        crate::gc_hook::try_gc_write_barrier(new_block as crate::gc_hook::GCREF);
    }
    // rlist.py `_ll_list_resize_hint_really`: `rgc.ll_arraycopy` then
    // `l.items = newitems`. The copy's `copy_item` head is what writeanalyze
    // records as this ARRAY's read and write, so `force_from_effectinfo`
    // flushes lazy SETARRAYITEM_GC before the residual COND_CALL.
    if !old.is_null() && live_len > 0 {
        unsafe {
            ll_arraycopy_items_block(
                crate::gc_roots::shadow_stack_get(old_slot) as *mut ItemsBlock,
                crate::gc_roots::shadow_stack_get(new_block_slot) as *mut ItemsBlock,
                0,
                0,
                live_len as i64,
            );
        }
    }
    let new_block = crate::gc_roots::shadow_stack_get(new_block_slot) as *mut ItemsBlock;
    let new_base = unsafe { items_block_items_base(new_block) };
    for i in live_len..new_cap {
        unsafe { *new_base.add(i) = PY_NULL };
    }
    Some(new_block)
}

/// Tuple-construction allocator on the Phase L2 nursery path. Exact-size
/// (`cap == len` — tuples are immutable and every slot is written, no
/// overallocation). Fills from `cap` already-pinned shadow-stack slots at
/// `save_point`, which the caller must keep live across this call: a
/// collection during the block malloc rewrites those slots in place, and
/// the fill reads them back afterwards. A `Vec` of the same words would
/// still hold pre-move addresses. Degrades to `std::alloc` when the gate
/// is off, still filling from the slots.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_tuple_items_block_gc(save_point: usize, cap: usize) -> *mut ItemsBlock {
    if !itemsblock_gc_enabled() {
        let block = unsafe { alloc_items_block(cap) };
        if cap > 0 {
            let dst = unsafe { std::slice::from_raw_parts_mut(items_block_items_base(block), cap) };
            crate::gc_roots::shadow_stack_copy_range(save_point, dst);
        }
        return block;
    }
    let _roots = crate::gc_roots::push_roots();
    let block_slot = crate::gc_roots::shadow_stack_len();
    let (block, owns_block) = unsafe { alloc_items_block_gc(cap) };
    let _ = crate::gc_roots::pin_root(block as PyObjectRef);
    // Barrier before the fill — see `alloc_list_items_block_gc`.
    // `owns_block` is the placement `try_alloc_items_block_gc` already knew.
    let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
    let base = unsafe { items_block_items_base(block) };
    if cap > 0 {
        // pop_roots of the caller's item livevars, not a snapshot taken
        // before this malloc.
        let dst = unsafe { std::slice::from_raw_parts_mut(base, cap) };
        crate::gc_roots::shadow_stack_copy_range(save_point, dst);
    }
    // The block may have landed in old-gen (nursery-full fallback) while
    // its elements are still young. `write_barrier` /
    // `remember_young_pointer` is not a collection point, so the fill
    // can precede it; the managed entry is the one every other typed
    // array uses after a `setarrayitem` into a `malloc_varsize` block.
    // A nursery block carries no TRACK_YOUNG_PTRS and the barrier is a
    // no-op. Guard on GC ownership exactly like `list_write_barrier`.
    if owns_block {
        crate::gc_hook::try_gc_write_barrier_managed(
            crate::gc_roots::shadow_stack_get(block_slot) as crate::gc_hook::GCREF
        );
    }
    crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock
}

/// Allocate an exact-`cap` NULL-filled GC-managed `ItemsBlock` of refs for
/// the walker's recording-time materialization of a virtual
/// `NEW_ARRAY_CLEAR` (BUILD_LIST / BUILD_TUPLE). Returns `None` when the gate
/// is off or no GC hook is installed, so the caller declines materialization
/// rather than handing back a `std::alloc` block that would never be freed.
/// The block is a `PY_OBJECT_ARRAY_GC_TYPE_ID` varsize array, so its items are
/// GC-traced and the NULL prefix is benign until the caller fills slots via
/// `setarrayitem_ref` + a block write barrier. No `std::alloc` fallback here
/// (unlike [`alloc_items_block_gc`]): the materialization requires a GC-traced
/// block whose escaping young element refs are forwarded by a collection.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_cleared_ref_items_block_gc(cap: usize) -> Option<*mut ItemsBlock> {
    if !itemsblock_gc_enabled() {
        return None;
    }
    let payload = ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<PyObjectRef>();
    let raw = crate::gc_hook::try_gc_alloc(PY_OBJECT_ARRAY_GC_TYPE_ID, payload)?;
    if raw.is_null() {
        return None;
    }
    let block = raw as *mut ItemsBlock;
    unsafe {
        (*block).capacity = cap;
        let base = items_block_items_base(block);
        for i in 0..cap {
            *base.add(i) = PY_NULL;
        }
    }
    Some(block)
}

/// Allocate a fresh `ItemsBlock` with the given capacity via
/// `std::alloc::alloc`. This is the `MAJIT_GC_ITEMSBLOCK=0` fallback;
/// the default path is the moving-nursery `alloc_items_block_gc` above
/// (`try_gc_alloc`, caller-side root tracking via `gc_roots::pin_root` +
/// the block write-barrier). An earlier `try_gc_alloc_stable` attempt was
/// abandoned because old-gen (non-moving) allocation accumulates
/// per-iteration containers until a major GC; the nursery path avoids that
/// (minor GC reclaims short-lived blocks) and is perf-neutral on
/// fannkuch/nbody.
///
/// The capacity header is initialized; items are left uninitialized —
/// the caller must write all `capacity` slots before exposing the
/// pointer to the GC walker. `cap` may be zero; the resulting block
/// holds only the 8-byte capacity header (used by tuple — see
/// [`alloc_tuple_items_block`] — for empty tuples).
unsafe fn alloc_items_block(cap: usize) -> *mut ItemsBlock {
    unsafe { try_alloc_items_block(cap).unwrap_or_else(|| items_block_alloc_failed(cap)) }
}

/// [`alloc_items_block`] for a capacity that came from Python, where the
/// caller has a `MemoryError` to report instead of the abort above.
unsafe fn try_alloc_items_block(cap: usize) -> Option<*mut ItemsBlock> {
    let layout = try_items_block_layout(cap)?;
    unsafe {
        let block = std_alloc_gc_array(layout, false)? as *mut ItemsBlock;
        (*block).capacity = cap;
        Some(block)
    }
}

/// `gcheaderbuilder.size_gc_header`: the `GcHeader` prefix of a GcArray
/// block, spelled as the `size_of` / `align_of` the translator folds to a
/// constant.
const GC_ARRAY_HEADER_SIZE: usize = std::mem::size_of::<majit_gc::header::GcHeader>();
const GC_ARRAY_HEADER_ALIGN: usize = std::mem::align_of::<majit_gc::header::GcHeader>();

/// Byte size of the `[GcHeader | payload]` block of a no-hook `std::alloc`
/// GcArray, or `None` when no layout of the header's alignment describes it:
/// `Layout::from_size_align` requires the size rounded up to the alignment to
/// stay within `isize`.
fn std_gc_array_size(payload: Layout) -> Option<usize> {
    let size = GC_ARRAY_HEADER_SIZE.checked_add(payload.size())?;
    let rounded = size.checked_add(GC_ARRAY_HEADER_ALIGN - 1)?;
    if (rounded as isize) < 0 {
        None
    } else {
        Some(size)
    }
}

/// `std::alloc` layout of a GcArray block of `size` bytes, a size
/// [`std_gc_array_size`] accepted. The block takes the header's alignment: no
/// GcArray payload is aligned more strictly (asserted below), so the constant
/// is the maximum of the two.
fn std_gc_array_layout(size: usize) -> Layout {
    Layout::from_size_align(size, GC_ARRAY_HEADER_ALIGN).expect("std GcArray layout")
}

// The payload starts `GcHeader::SIZE` bytes into the allocation, so the header
// size must keep every block's payload alignment, and the block's alignment is
// the header's only while no payload needs more.
const _: () = assert!(
    GC_ARRAY_HEADER_SIZE == majit_gc::header::GcHeader::SIZE
        && GC_ARRAY_HEADER_ALIGN == majit_gc::header::GcHeader::ALIGN
        && GC_ARRAY_HEADER_SIZE % std::mem::align_of::<ItemsBlock>() == 0
        && GC_ARRAY_HEADER_SIZE % std::mem::align_of::<TypedItemsBlock>() == 0
        && GC_ARRAY_HEADER_SIZE % std::mem::align_of::<FixedObjectArray>() == 0
        && std::mem::align_of::<ItemsBlock>() <= GC_ARRAY_HEADER_ALIGN
        && std::mem::align_of::<TypedItemsBlock>() <= GC_ARRAY_HEADER_ALIGN
        && std::mem::align_of::<FixedObjectArray>() <= GC_ARRAY_HEADER_ALIGN,
    "GcHeader must preserve and bound the GcArray payload alignment",
);

/// The no-hook `std::alloc` fallback for a GcArray: every GcArray carries a GC
/// header, so the block is `[GcHeader | payload]` with a zeroed header, the
/// `alloc_fixed_array_with_header` shape. The zero flag word keeps
/// `TRACK_YOUNG_PTRS` clear, so the inline flag test in `setarrayitem_gc`
/// (`items_block_set_ref`, [`FixedObjectArray::set_ref`]) takes the plain
/// store for these blocks with no ownership query. Returns the payload
/// address, or `None` when the allocator refuses.
unsafe fn std_alloc_gc_array(payload: Layout, zeroed: bool) -> Option<*mut u8> {
    let layout = std_gc_array_layout(std_gc_array_size(payload)?);
    unsafe {
        let raw = if zeroed {
            alloc_zeroed(layout)
        } else {
            alloc(layout)
        };
        if raw.is_null() {
            return None;
        }
        std::ptr::write_bytes(raw, 0, GC_ARRAY_HEADER_SIZE);
        // `cast_adr_to_ptr(result + size_gc_header)`: the payload address is
        // header address arithmetic, as `header_of` spells the inverse.
        Some((raw as usize + GC_ARRAY_HEADER_SIZE) as *mut u8)
    }
}

/// Free a block [`std_alloc_gc_array`] returned for the same `payload` layout.
unsafe fn std_dealloc_gc_array(block: *mut u8, payload: Layout) {
    let size = std_gc_array_size(payload).expect("std GcArray layout");
    let header = (block as usize - GC_ARRAY_HEADER_SIZE) as *mut u8;
    unsafe { dealloc(header, std_gc_array_layout(size)) };
}

/// The abort an infallible `ItemsBlock` allocation takes, so the fallible
/// twins have one place to send a `None` for a caller with no failure edge.
///
/// A capacity whose layout does not even exist is reported against the bare
/// header, the same stand-in [`alloc_typed_items_block`] uses.
pub fn items_block_alloc_failed(cap: usize) -> ! {
    std::alloc::handle_alloc_error(
        try_items_block_layout(cap).unwrap_or_else(|| Layout::new::<ItemsBlock>()),
    )
}

/// `rlist.py _ll_prebuilt_empty_array` (`specialize:memo`): the one
/// zero-length `GcArray(OBJECTPTR)` every empty resizable list's `items`
/// points at (`ll_newemptylist`), so `len(l.items)` is readable on a list
/// that holds nothing.  `OnceLock<usize>` is the process-global immortal
/// owner the prebuilt objects use.
static PREBUILT_EMPTY_ITEMS_BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// The prebuilt empty `items` block.  The body reads the static, which the
/// translator resolves to the registered prebuilt address, so a traced call
/// is that constant.
///
/// The block is [`std_alloc_gc_array`]'s shape with capacity 0: a zeroed
/// header, so no store barrier fires for it, and no slot, so nothing is ever
/// stored.  It is never freed ([`dealloc_items_block`] passes it by).
#[inline]
pub fn ll_prebuilt_empty_items_block() -> *mut ItemsBlock {
    *PREBUILT_EMPTY_ITEMS_BLOCK.get_or_init(|| {
        // SAFETY: the layout is the bare `ItemsBlock` header.
        unsafe { std_alloc_gc_array(items_block_layout(0), true) }
            .unwrap_or_else(|| items_block_alloc_failed(0)) as usize
    }) as *mut ItemsBlock
}

/// Deallocate an `ItemsBlock` previously allocated via
/// [`alloc_items_block`] or [`grow_items_block`]. Phase L2: a
/// GC-managed block (nursery / old-gen) is reclaimed by the collector
/// and must never be freed here; only the header-prefixed `std::alloc`
/// fallback ([`std_alloc_gc_array`]) goes back to the allocator.
/// `try_gc_owns_object` discriminates the two block origins.
unsafe fn dealloc_items_block(block: *mut ItemsBlock) {
    if block.is_null() || block == ll_prebuilt_empty_items_block() {
        return;
    }
    if crate::gc_hook::try_gc_owns_object(block as crate::gc_hook::GCREF) {
        return;
    }
    unsafe {
        let cap = (*block).capacity;
        std_dealloc_gc_array(block as *mut u8, items_block_layout(cap));
    }
}

fn items_block_layout(cap: usize) -> Layout {
    try_items_block_layout(cap).expect("ItemsBlock layout")
}

/// [`items_block_layout`] with the size arithmetic checked: a capacity that
/// came from Python can overflow the multiply, which wraps in release and
/// hands back a layout far smaller than the block the caller then writes.
fn try_items_block_layout(cap: usize) -> Option<Layout> {
    let items = cap.checked_mul(std::mem::size_of::<PyObjectRef>())?;
    let total = ITEMS_BLOCK_ITEMS_OFFSET.checked_add(items)?;
    Layout::from_size_align(total, std::mem::align_of::<ItemsBlock>()).ok()
}

/// Return the items base pointer of an `ItemsBlock`.
#[inline]
unsafe fn items_block_items_ptr(block: *mut ItemsBlock) -> *mut PyObjectRef {
    unsafe { (block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET) as *mut PyObjectRef }
}

/// Reallocate an `ItemsBlock` to a new capacity, copying live items.
/// Spare slots `live_len..capacity` are NULL-initialized so the GC
/// walker (once `PY_OBJECT_ARRAY_GC_TYPE_ID` is active on the
/// allocation) sees valid NULL refs in unused slots — upstream
/// relies on `gc_malloc_array` zero-filling the fresh block
/// (rlist.py `_ll_list_resize_really`); pyre's `alloc`
/// uses `std::alloc::alloc` which is not zero-filled so we
/// explicit-init here.
/// Old block is deallocated. Returns the new block.
unsafe fn grow_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    unsafe {
        try_grow_items_block(old, new_cap, live_len)
            .unwrap_or_else(|| items_block_alloc_failed(new_cap))
    }
}

/// [`grow_items_block`] for a capacity that came from Python.  The fresh block
/// is the only fallible step and nothing is published before it, so a refusal
/// leaves `old` exactly as it was.
unsafe fn try_grow_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> Option<*mut ItemsBlock> {
    unsafe {
        let fresh = try_alloc_items_block(new_cap)?;
        let new_base = items_block_items_ptr(fresh);
        if !old.is_null() && live_len > 0 {
            std::ptr::copy_nonoverlapping(items_block_items_ptr(old), new_base, live_len);
        }
        let fresh_cap = (*fresh).capacity;
        for i in live_len..fresh_cap {
            *new_base.add(i) = PY_NULL;
        }
        Some(fresh)
    }
}

// ─── TypedItemsBlock: GcArray(Float)/GcArray(Signed) backing block ───────
//
// The block itself is `majit_rlib::lltypesystem::rlist`, re-exported at the top
// of this file: the Float / Integer list strategies (`listobject.py`
// FloatListStrategy / IntegerListStrategy) and `rbigint._digits` are the same
// `erase([float])` / `erase([int])` lowering, so one body serves all three.
//
// What remains here is the stable (old-gen) tier the list strategies allocate
// in, which roots through this crate's shadow stack, plus the array tokens the
// descr registry reads.

/// `get_array_token(GcArray(Signed))` — [`gc_int_array_gc_type_id`].
pub const TYPED_ITEMS_BLOCK_INT_TOKEN: ArrayToken = ArrayToken {
    base_size: TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: std::mem::size_of::<i64>(),
    len_offset: TYPED_ITEMS_BLOCK_LEN_OFFSET,
};

/// `get_array_token(GcArray(Float))` — [`gc_float_array_gc_type_id`]. A distinct
/// ARRAY identity from the signed one, sharing this block's layout.
pub const TYPED_ITEMS_BLOCK_FLOAT_TOKEN: ArrayToken = ArrayToken {
    base_size: TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: std::mem::size_of::<f64>(),
    len_offset: TYPED_ITEMS_BLOCK_LEN_OFFSET,
};

/// The items are `u64` words, so an element type wider or narrower than the
/// storage word would stride past or short of them.
const _: () = assert!(
    TYPED_ITEMS_BLOCK_INT_TOKEN.item_size == std::mem::size_of::<u64>()
        && TYPED_ITEMS_BLOCK_FLOAT_TOKEN.item_size == std::mem::size_of::<u64>(),
    "TypedItemsBlock stores u64 words; another element width needs its own block",
);

/// Allocate a fresh zero-filled `TypedItemsBlock` with the given capacity, as a
/// `tid` (`GC_INT_ARRAY` / `GC_FLOAT_ARRAY`) varsize GcArray. Zero-fill matches
/// `gc_malloc_array` (rlist.py `_ll_list_resize_really`) and the
/// Float/Int strategy `_none_value` (0.0 / 0); `try_gc_alloc_stable` memory is
/// not guaranteed zeroed, so the items are cleared explicitly. `cap` is clamped
/// to at least 1 (rlist.py:251 overallocation policy). Falls back to
/// `std::alloc::alloc_zeroed` when the gate is off or no GC hook is installed
/// (pure interpreter / early startup).
///
/// The block is allocated `stable` (old-gen, mark-sweep, non-moving) — the same
/// tier as its `W_ListObject` owner (`listobject.rs` `try_gc_alloc_stable`).
/// The items are plain scalars (no GC pointers), so the block is a varsize leaf
/// the collector marks: it never relocates and never holds a young pointer, so
/// the owning list needs no remembered-set barrier when the block changes. Its
/// `int_items.block` / `float_items.block` owner slot is marked-live by
/// `list_object_custom_trace` — but only once that slot holds the block. Being
/// old-gen buys immobility, not survival: a block born before this cycle
/// finished scanning carries no `GCFLAG_VISITED`, so between this call and the
/// owner store the caller must root it (`IntArray::pin_block`).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_typed_items_block(cap: usize, tid: u32) -> *mut TypedItemsBlock {
    unsafe {
        try_alloc_typed_items_block(cap, tid)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(Layout::new::<TypedItemsBlock>()))
    }
}

/// Fallible `gc_malloc_array` surface used where the RPython caller exposes
/// allocation failure as `MemoryError` (notably `rbigint.lshift`).  The
/// ordinary translated allocation path above remains infallible at the Rust
/// type level, matching RPython's implicit exception edge; this entry point
/// makes that edge explicit for Rust methods that already return `Result`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn try_alloc_typed_items_block(cap: usize, tid: u32) -> Option<*mut TypedItemsBlock> {
    let cap = cap.max(1);
    let layout = try_typed_items_block_layout(cap)?;
    let payload = layout.size();
    if itemsblock_gc_enabled() {
        match crate::gc_hook::GcAllocOutcome::from_hook(crate::gc_hook::try_gc_alloc_stable(
            tid, payload,
        )) {
            crate::gc_hook::GcAllocOutcome::Allocated(raw) => {
                let block = raw as *mut TypedItemsBlock;
                unsafe {
                    (*block).capacity = cap;
                    std::ptr::write_bytes(
                        typed_items_block_items_base(block),
                        0,
                        cap * std::mem::size_of::<u64>(),
                    );
                }
                return Some(block);
            }
            // `try_gc_alloc_stable_raw` ends the process on this answer,
            // because its callers meet a null by allocating outside the heap
            // and a block holding managed references must never live there.  A
            // typed block holds unboxed words and its capacity comes from
            // Python, so the refusal is reported instead: nothing is allocated
            // anywhere on this path and the caller raises `MemoryError`.
            crate::gc_hook::GcAllocOutcome::Failed => return None,
            // No heap owns the request -- before the GC is up, or in a unit
            // test -- so the plain allocation below is the whole heap.
            crate::gc_hook::GcAllocOutcome::NoRoute => {}
        }
    }
    unsafe {
        let block = std_alloc_gc_array(layout, true)? as *mut TypedItemsBlock;
        (*block).capacity = cap;
        Some(block)
    }
}

/// `Ptr(GcArray(Float))` as a result type: the layout of [`TypedItemsBlock`]
/// with `f64` items. A malloc's ARRAY is its result's `concretetype.TO`
/// (jtransform.py `do_fixed_newlist`), and `TypedItemsBlock` names
/// `GcArray(Signed)`, so the Float malloc leaf returns this.
#[repr(C)]
pub struct FloatItemsBlock {
    pub capacity: usize,
    items: [f64; 0],
}

const _: () = assert!(
    std::mem::size_of::<FloatItemsBlock>() == std::mem::size_of::<TypedItemsBlock>()
        && std::mem::offset_of!(FloatItemsBlock, items) == TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
    "FloatItemsBlock and TypedItemsBlock are one layout",
);

/// rlist.py `_ll_list_resize_hint_really` `newitems = malloc(ITEMS, some)` for
/// `GcArray(Signed)`; `do_fixed_newlist` rewrites the call to `new_array`.
///
/// The body allocates the block old and non-moving, so the owner store that
/// follows needs no remembered-set barrier outside the JIT.
#[inline(never)]
#[majit_macros::oopspec("newlist(length)")]
pub(crate) unsafe fn ll_new_int_items(count: i64) -> *mut TypedItemsBlock {
    unsafe { alloc_typed_items_block(count as usize, gc_int_array_gc_type_id()) }
}

/// [`ll_new_int_items`] for `GcArray(Float)`.
#[inline(never)]
#[majit_macros::oopspec("newlist(length)")]
pub(crate) unsafe fn ll_new_float_items(count: i64) -> *mut FloatItemsBlock {
    unsafe {
        alloc_typed_items_block(count as usize, gc_float_array_gc_type_id()) as *mut FloatItemsBlock
    }
}

static PREBUILT_EMPTY_INT_ITEMS_BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static PREBUILT_EMPTY_FLOAT_ITEMS_BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn alloc_prebuilt_empty_typed_items_block() -> usize {
    unsafe {
        let block = alloc_typed_items_block_immortal(0);
        (*block).capacity = 0;
        block as usize
    }
}

/// rlist.py `_ll_prebuilt_empty_array(GcArray(Signed))` (`specialize:memo`):
/// the zero-length items array an empty Integer list holds, so
/// `len(l.items)` reads a real header.
#[inline]
pub fn ll_prebuilt_empty_int_items_block() -> *mut TypedItemsBlock {
    *PREBUILT_EMPTY_INT_ITEMS_BLOCK.get_or_init(|| alloc_prebuilt_empty_typed_items_block())
        as *mut TypedItemsBlock
}

/// [`ll_prebuilt_empty_int_items_block`] for `GcArray(Float)`.
#[inline]
pub fn ll_prebuilt_empty_float_items_block() -> *mut TypedItemsBlock {
    *PREBUILT_EMPTY_FLOAT_ITEMS_BLOCK.get_or_init(|| alloc_prebuilt_empty_typed_items_block())
        as *mut TypedItemsBlock
}

// rlist.py `_ll_list_resize_hint_really` is specialized per LIST type, so the
// int and float grows are two graphs, each copying through its own ARRAY's
// `ll_arraycopy`.
macro_rules! typed_items_block_grow {
    ($grow:ident, $try_grow:ident, $new:ident, $block:ty, $tid:path, $arraycopy:ident) => {
        /// rlist.py `_ll_list_resize_hint_really`: `newitems = malloc(...)`,
        /// `rgc.ll_arraycopy(items, newitems, 0, 0, p)`, and the caller's
        /// `l.items = newitems`. `old` may be null or the prebuilt empty
        /// block; the collector reclaims it once the owner stops naming it.
        ///
        /// `old` is a livevar across the malloc, so it is pinned and reloaded
        /// (`framework.py gct_fv_gc_malloc`); the copy's `copy_item` head is
        /// what writeanalyze records as this ARRAY's read and write.
        /// # Safety
        /// The caller must uphold every validity, runtime-type, aliasing, and
        /// lifetime invariant required by the object and pointer arguments for
        /// the entire call.
        pub unsafe fn $grow(
            old: *mut TypedItemsBlock,
            new_cap: usize,
            live_len: usize,
        ) -> *mut TypedItemsBlock {
            unsafe {
                let roots = crate::gc_roots::push_roots();
                let base = roots.base();
                let _ = roots.pin_root(old as crate::PyObjectRef);
                let newitems = $new(new_cap as i64) as *mut TypedItemsBlock;
                // Nothing on the heap names `newitems` until the caller's
                // `l.items = newitems`, so it stays rooted across the release
                // below as well as the copy.
                let _ = roots.pin_root(newitems as crate::PyObjectRef);
                if live_len > 0 {
                    $arraycopy(
                        roots.get(base) as *mut $block,
                        roots.get(base + 1) as *mut $block,
                        0,
                        0,
                        live_len as i64,
                    );
                }
                // A block outside the collector's heap (no GC installed, or
                // the items-block gate off) has no other owner to free it.
                // Traced code allocates its blocks from the collector, so
                // this arm is the interpreter's alone.
                if !majit_rlib::jit::we_are_jitted() {
                    dealloc_typed_items_block(roots.get(base) as *mut TypedItemsBlock);
                }
                roots.get(base + 1) as *mut TypedItemsBlock
            }
        }

        /// The grow for a capacity that came from Python.  The fresh block is
        /// the first step and the only fallible one, so a refusal leaves `old`
        /// allocated and still owned by its list.
        /// # Safety
        /// The caller must uphold every validity, runtime-type, aliasing, and
        /// lifetime invariant required by the object and pointer arguments for
        /// the entire call.
        pub unsafe fn $try_grow(
            old: *mut TypedItemsBlock,
            new_cap: usize,
            live_len: usize,
        ) -> Option<*mut TypedItemsBlock> {
            unsafe {
                let fresh = try_alloc_typed_items_block(new_cap, $tid())?;
                // `fresh` has no owner field yet, so nothing on the heap names
                // it for the span below. Old-gen is mark-sweep and a block born
                // outside `GcState::Marking` carries no `VISITED`, so the next
                // cycle sweeps an unreachable one — root it for the span, as
                // `gct_fv_gc_malloc` roots every livevar it brackets a malloc
                // with (`framework.py gct_fv_gc_malloc`).
                let _roots = crate::gc_roots::push_roots();
                let fresh_root = crate::gc_roots::shadow_stack_len();
                let _ = crate::gc_roots::pin_root(fresh as crate::PyObjectRef);
                // rlist.py `_ll_list_resize_hint_really`: `rgc.ll_arraycopy`
                // then `l.items = newitems`. The copy's `copy_item` head is
                // what writeanalyze records as this ARRAY's read and write, so
                // `force_from_effectinfo` flushes lazy SETARRAYITEM_GC before
                // the residual COND_CALL.
                // A null `old` is pinned too, so every path publishes the
                // same slots.
                let old_root = crate::gc_roots::shadow_stack_len();
                let _ = crate::gc_roots::pin_root(old as crate::PyObjectRef);
                if !old.is_null() && live_len > 0 {
                    $arraycopy(
                        crate::gc_roots::shadow_stack_get(old_root) as *mut $block,
                        crate::gc_roots::shadow_stack_get(fresh_root) as *mut $block,
                        0,
                        0,
                        live_len as i64,
                    );
                }
                if !old.is_null() {
                    dealloc_typed_items_block(
                        crate::gc_roots::shadow_stack_get(old_root) as *mut TypedItemsBlock
                    );
                }
                Some(crate::gc_roots::shadow_stack_get(fresh_root) as *mut TypedItemsBlock)
            }
        }
    };
}

typed_items_block_grow!(
    grow_int_items_block,
    try_grow_int_items_block,
    ll_new_int_items,
    TypedItemsBlock,
    gc_int_array_gc_type_id,
    ll_arraycopy_int_items_block
);
typed_items_block_grow!(
    grow_float_items_block,
    try_grow_float_items_block,
    ll_new_float_items,
    FloatItemsBlock,
    gc_float_array_gc_type_id,
    ll_arraycopy_float_items_block
);

/// Deallocate a `TypedItemsBlock`. No-op on null. A GC-managed block is reclaimed
/// by the collector and must never be freed here. `try_gc_owns_object` gates the
/// `std::alloc` free to the gate-off / no-hook fallback blocks
/// ([`std_alloc_gc_array`]).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn dealloc_typed_items_block(block: *mut TypedItemsBlock) {
    if block.is_null()
        || block == ll_prebuilt_empty_int_items_block()
        || block == ll_prebuilt_empty_float_items_block()
    {
        return;
    }
    if crate::gc_hook::try_gc_owns_object(block as crate::gc_hook::GCREF) {
        return;
    }
    unsafe {
        let cap = (*block).capacity;
        std_dealloc_gc_array(block as *mut u8, typed_items_block_layout(cap));
    }
}

// ─── FixedObjectArray: pyframe.py:112 make_sure_not_resized parity ────────
//
// RPython `locals_cells_stack_w = [None] * size; make_sure_not_resized(...)`
// becomes a fixed-length GcArray (`Ptr(GcArray(PyObjectRef))`). The layout
// here matches that upstream shape so `GETFIELD_GC_R(frame, locals_cells_stack)
// + GETARRAYITEM_GC_R(array_ptr, i)` means the same thing in both worlds:
// single-indirection, items immediately after the length header.
//
// Layout: `[len: usize] [items: PyObjectRef; len]` (variable-length,
// flexible-array tail). Allocation happens via a custom `Layout` at the
// caller site (see `pyre_interpreter::pyframe::alloc_fixed_array_with_header`).

/// Offset of the length prefix within `FixedObjectArray`.
pub const FIXED_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(FixedObjectArray, len);

/// Offset of the first item within `FixedObjectArray` (immediately after
/// the length prefix). The JIT-visible array descriptor uses this as
/// `base_size` so `GETARRAYITEM_GC_*` reads items directly.
pub const FIXED_ARRAY_ITEMS_OFFSET: usize = std::mem::offset_of!(FixedObjectArray, _items);

/// `get_array_token(GcArray(PyObjectRef))` for the frame-locals and mro blocks.
pub const FIXED_OBJECT_ARRAY_TOKEN: ArrayToken = ArrayToken {
    base_size: FIXED_ARRAY_ITEMS_OFFSET,
    item_size: std::mem::size_of::<PyObjectRef>(),
    len_offset: FIXED_ARRAY_LEN_OFFSET,
};

/// `FixedObjectArray` blocks are allocated under [`PY_OBJECT_ARRAY_GC_TYPE_ID`]
/// as well (`pyframe::alloc_frame_locals_array`, [`alloc_mro_block_gc`]), and one
/// tid carries one varsize shape — the one registered from [`ITEMS_BLOCK_TOKEN`].
/// Both structs are a length word followed by `[PyObjectRef; 0]` today, so the
/// two shapes coincide on every target. Diverge them and the collector would
/// size one of the two wrong, which on a 32-bit target is a short copy rather
/// than a build error. Keep it a build error.
const _: () = assert!(
    FIXED_OBJECT_ARRAY_TOKEN.base_size == ITEMS_BLOCK_TOKEN.base_size
        && FIXED_OBJECT_ARRAY_TOKEN.item_size == ITEMS_BLOCK_TOKEN.item_size
        && FIXED_OBJECT_ARRAY_TOKEN.len_offset == ITEMS_BLOCK_TOKEN.len_offset,
    "FixedObjectArray and ItemsBlock share one GcArray tid; their shapes must match",
);

/// pyframe.py:110-112: fixed-length GcArray for `locals_cells_stack_w`.
///
/// Once created, the length never changes. No push, no grow.
/// Items are mutable (stack operations write via index) but the
/// array cannot be resized.
///
/// `_items` is a zero-sized flexible-array marker: the real items live
/// immediately after `len` in the allocation, accessed via pointer
/// arithmetic through `items_ptr` / `as_slice`.
#[repr(C)]
pub struct FixedObjectArray {
    /// Length prefix. Matches RPython `Ptr(GcArray(T))` header so that
    /// the JIT's arraydescr `base_size = FIXED_ARRAY_ITEMS_OFFSET` lands
    /// on items[0].
    pub len: usize,
    /// Flexible-array tail marker. Actual items follow immediately in
    /// memory (sized to `len` at allocation time); this field has size 0.
    _items: [PyObjectRef; 0],
}

impl FixedObjectArray {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn items_ptr(&self) -> *const PyObjectRef {
        unsafe {
            (self as *const Self as *const u8).add(FIXED_ARRAY_ITEMS_OFFSET) as *const PyObjectRef
        }
    }

    #[inline]
    pub fn items_mut_ptr(&mut self) -> *mut PyObjectRef {
        unsafe { (self as *mut Self as *mut u8).add(FIXED_ARRAY_ITEMS_OFFSET) as *mut PyObjectRef }
    }

    pub fn as_slice(&self) -> &[PyObjectRef] {
        unsafe { std::slice::from_raw_parts(self.items_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [PyObjectRef] {
        unsafe { std::slice::from_raw_parts_mut(self.items_mut_ptr(), self.len) }
    }

    /// Store a GC reference through the host interpreter's equivalent of the
    /// GC transform's `setarrayitem_gc` rewrite.
    ///
    /// `transform_generic_set` emits the inline `TRACK_YOUNG_PTRS` test, the
    /// conditional array write barrier, then the store. The barrier is not a
    /// collection point, so `value` needs no root across it.
    #[inline]
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
    )]
    pub fn set_ref(&mut self, index: usize, value: PyObjectRef) {
        assert!(index < self.len);
        // Clearing a slot cannot create an old-to-young edge.  PyPy's
        // `popvalue_maybe_none` / `dropvalues*` therefore compile to the
        // plain `None` store with no collecting write-barrier slow path.
        if value.is_null() {
            unsafe { self.items_mut_ptr().add(index).write(value) };
            return;
        }
        // The ordinary `setarrayitem_gc` rewrite: inspect TRACK_YOUNG_PTRS
        // inline and enter the barrier slow path only while the old array
        // still needs to be remembered.  The barrier clears this bit, so every
        // later store into the same array is a plain write.  Nursery arrays and
        // the header-prefixed `std::alloc` fallback arrays carry a zero flag
        // word and take this arm.
        let header = unsafe { majit_gc::header::header_of(self as *mut Self as usize) };
        if unsafe { !(*header).has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            unsafe { self.items_mut_ptr().add(index).write(value) };
            return;
        }
        self.set_ref_remembering(index, value);
    }

    /// The flagged half of [`Self::set_ref`], kept out of line the way the GC
    /// transform inlines only the `TRACK_YOUNG_PTRS` test of `setarrayitem_gc`
    /// and calls `remember_young_pointer_from_array` for the rest.
    #[cold]
    #[inline(never)]
    fn set_ref_remembering(&mut self, index: usize, value: PyObjectRef) {
        let header = unsafe { majit_gc::header::header_of(self as *mut Self as usize) };
        // `FORWARDED_MARKER` sets every flag bit, so `set_ref`'s flag test cannot
        // tell an array a minor collection already moved from a live old one.
        // Report it here, where the array and the store that reached it are
        // both still named, instead of one frame deeper where the barrier sees
        // only a type id past the end of the type table.
        if unsafe { (*header).is_forwarded() } {
            stale_array_abort(self as *mut Self as usize, index);
        }
        // incminimark.py `write_barrier` / `remember_young_pointer`:
        // append to `old_objects_pointing_to_young` and clear
        // `TRACK_YOUNG_PTRS`. That helper is not a collection point;
        // `setarrayitem_gc` does not `push_roots` around it. A pin here
        // used to wait on `try_gc_current_object_address` and consume
        // the birth remembered-set entry mid-fill, after which later
        // stores took the no-barrier arm.
        crate::gc_hook::try_gc_write_barrier_managed(self as *mut Self as crate::gc_hook::GCREF);
        unsafe { self.items_mut_ptr().add(index).write(value) };
    }

    /// Atomic load of slot `index`. The lazy `co_consts_w` first-fill
    /// pairs this with [`Self::compare_exchange_ref`].
    #[inline]
    pub fn load_atomic(&self, index: usize, order: Ordering) -> PyObjectRef {
        assert!(index < self.len);
        self.item_atomic_ptr(index).load(order)
    }

    /// Remember the array, then CAS slot `index`. The barrier is
    /// `remember_young_pointer` and is not a collection point; it runs
    /// before the publish so a concurrent minor that already holds the
    /// table in the remembered set still traces the new young child.
    #[inline]
    pub fn compare_exchange_ref(
        &self,
        index: usize,
        current: PyObjectRef,
        new: PyObjectRef,
        success: Ordering,
        failure: Ordering,
    ) -> Result<PyObjectRef, PyObjectRef> {
        assert!(index < self.len);
        if !new.is_null() {
            let header = unsafe { majit_gc::header::header_of(self as *const Self as usize) };
            if unsafe { (*header).has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
                if unsafe { (*header).is_forwarded() } {
                    stale_array_abort(self as *const Self as usize, index);
                }
                crate::gc_hook::try_gc_write_barrier_managed(
                    self as *const Self as crate::gc_hook::GCREF,
                );
            }
        }
        self.item_atomic_ptr(index)
            .compare_exchange(current, new, success, failure)
    }

    #[inline]
    fn item_atomic_ptr(&self, index: usize) -> &AtomicPtr<PyObject> {
        unsafe { &*self.items_ptr().add(index).cast::<AtomicPtr<PyObject>>() }
    }

    pub fn to_vec(&self) -> Vec<PyObjectRef> {
        self.as_slice().to_vec()
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b);
    }
}

/// Out-of-line half of [`FixedObjectArray::set_ref`]'s forwarded-header check.
///
/// A forwarded header means the array is a nursery corpse: the collection that
/// moved it left this holder's field naming the old copy.  Print the live copy
/// and hand the address to whoever can name the holder before aborting, so the
/// surviving question is only which field kept the pre-collection pointer.
#[cold]
#[inline(never)]
fn stale_array_abort(array_addr: usize, index: usize) -> ! {
    let header = unsafe { majit_gc::header::header_of(array_addr) };
    let moved_to = unsafe { majit_gc::header::GcHeader::forwarding_address(header) };
    eprintln!(
        "STALE ARRAY: FixedObjectArray::set_ref(index={index}) reached {array_addr:#x}, \
         already moved to {moved_to:#x}"
    );
    // The holder is not on the live frame chain in any observed occurrence, so
    // the walk below cannot name it.  Capture the Rust stack here instead: this
    // arm is `#[cold]` and ends the process, so the cost is irrelevant and the
    // return addresses are the only thing that names the caller that kept the
    // pre-collection pointer.
    eprintln!(
        "STALE ARRAY backtrace:\n{}",
        std::backtrace::Backtrace::force_capture()
    );
    let named = crate::gc_hook::report_stale_array_holder(array_addr);
    panic!(
        "FixedObjectArray::set_ref on {array_addr:#x}: a minor collection moved this \
         array to {moved_to:#x} and its holder kept the pre-collection pointer \
         (holder named: {named})"
    );
}

impl Index<usize> for FixedObjectArray {
    type Output = PyObjectRef;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for FixedObjectArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

fn mro_block_layout(len: usize) -> Layout {
    Layout::from_size_align(
        FIXED_ARRAY_ITEMS_OFFSET + len * std::mem::size_of::<PyObjectRef>(),
        std::mem::align_of::<FixedObjectArray>(),
    )
    .expect("mro block layout")
}

/// Free an [`alloc_mro_block_gc`] block. A GC-owned block is the collector's;
/// only the no-hook `std::alloc` fallback is returned to the allocator. No-op
/// on null.
/// # Safety
/// `block` is null or a block [`alloc_mro_block_gc`] returned, not yet freed.
pub unsafe fn dealloc_mro_block(block: *mut FixedObjectArray) {
    if block.is_null() || crate::gc_hook::try_gc_owns_object(block as crate::gc_hook::GCREF) {
        return;
    }
    unsafe {
        let len = (*block).len;
        std_dealloc_gc_array(block as *mut u8, mro_block_layout(len));
    }
}

/// Allocate a fixed-length `FixedObjectArray` GcArray pre-filled from
/// `values`, for `W_TypeObject.mro_w` (typeobject.py `mro_w?[*]`, an
/// immutable `[W_Root]`). The block is `PY_OBJECT_ARRAY_GC_TYPE_ID`
/// (`Ptr(GcArray(OBJECTPTR))`), so the collector reads its length from the
/// offset-0 header and forwards items[0..len]; the prepass types
/// `w_type_get_mro`'s result as `SomeList(SomeInstance(PyObjectRef))`.
///
/// Allocated on the **stable** (non-moving old-gen) path via
/// [`crate::gc_hook::try_gc_alloc_stable_raw`] — code-object const/name
/// tables share this helper and outlive a minor collection as old-gen
/// arrays. Heap-type `mro_w` uses [`alloc_mro_block_gc_young`] so the
/// type↔MRO cycle can die at the next minor.
/// Falls back to a header-prefixed `std::alloc` block when no GC hook
/// is installed (bootstrap / pure interpreter).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_mro_block_gc(values: &[PyObjectRef]) -> *mut FixedObjectArray {
    let payload = FIXED_ARRAY_ITEMS_OFFSET + std::mem::size_of_val(values);
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(PY_OBJECT_ARRAY_GC_TYPE_ID, payload);
    unsafe { fill_mro_block(values, raw) }
}

/// [`alloc_mro_block_gc`] for a heap type's `mro_w`: the same non-moving
/// block, born young (`external_malloc(..., alloc_young=True)`) so that it
/// dies with the type at the first minor collection that reaches neither.
/// The block holds the type itself (`mro_w[0]`), so an old-born block on
/// the remembered set would promote a class that nothing else refers to.
/// # Safety
/// Same as [`alloc_mro_block_gc`].
pub unsafe fn alloc_mro_block_gc_young(values: &[PyObjectRef]) -> *mut FixedObjectArray {
    let payload = FIXED_ARRAY_ITEMS_OFFSET + std::mem::size_of_val(values);
    let raw = crate::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(
        PY_OBJECT_ARRAY_GC_TYPE_ID,
        payload,
    );
    unsafe { fill_mro_block(values, raw) }
}

unsafe fn fill_mro_block(
    values: &[PyObjectRef],
    raw: crate::gc_hook::GCREF,
) -> *mut FixedObjectArray {
    let len = values.len();
    let block = if !raw.is_null() {
        raw as *mut FixedObjectArray
    } else {
        // Bootstrap / no-hook fallback: a header-prefixed `std::alloc` block
        // ([`std_alloc_gc_array`]); the collector does not own it.
        let layout = mro_block_layout(len);
        (unsafe { std_alloc_gc_array(layout, false) })
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
            as *mut FixedObjectArray
    };
    // The ownership question runs before the elements are written rather than
    // between them and the barrier — see `alloc_list_items_block_gc`.  This
    // block is raw-malloced and never moves, so no re-read is owed for the
    // address.
    let owns_block = crate::gc_hook::try_gc_owns_object(block as crate::gc_hook::GCREF);
    unsafe {
        (*block).len = len;
        // Old→young barrier if the block landed in old-gen and any element is a
        // young object: registers the block on the remembered set so a later
        // minor collection forwards the young MRO entries.
        if owns_block {
            crate::gc_hook::try_gc_write_barrier(block as crate::gc_hook::GCREF);
        }
        let items = (*block).items_mut_ptr();
        let src = values.as_ptr();
        let mut i = 0usize;
        while i < len {
            items.add(i).write(*src.add(i));
            i += 1;
        }
    }
    block
}

// ─── GcTypedArray: typed array helper for resume / blackhole ─────────
//
// llmodel.py: bh_new_array / bh_new_array_clear
// llmodel.py: bh_setarrayitem_gc_r/i/f, bh_getarrayitem_gc_r/i/f
// resume.py: ResumeDataDirectReader allocate_array + setarrayitem_*
//
// RPython GC arrays are flat varsize blocks:
//
//     [length: WORD] [item_0] [item_1] ... [item_(length-1)]
//
// `GcTypedArray` mirrors that storage shape so backend
// `bh_getarrayitem_gc_*` / `bh_setarrayitem_gc_*` pointer arithmetic can
// operate on the same object that resume materialization returns.  This
// is still an allocator adaptation: the block is allocated with
// `std::alloc::alloc_zeroed` rather than MiniMark's nursery, so GC
// tracing/write barriers are not claimed here.

/// Offset of the length prefix within `GcTypedArray`.
pub const GC_TYPED_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(GcTypedArray, len);

/// Offset of the first item within `GcTypedArray`.
///
/// No [`ArrayToken`] companion, and deliberately so: the item size comes from
/// the resume / blackhole array descriptor rather than from this block (one
/// struct standing in for the Ref / Int / Float / Struct ARRAY lltypes), and no
/// tid is registered for it — `allocate_flat_gc_typed_array` allocates through
/// `std::alloc::alloc_zeroed`, so the collector never reads a shape here. These
/// two offsets describe this struct's own pointer arithmetic only; they are not
/// a GcArray shape any tid may be registered with.
pub const GC_TYPED_ARRAY_ITEMS_OFFSET: usize = std::mem::offset_of!(GcTypedArray, items);

/// Typed varsize array helper used by the resume / blackhole readers.
/// The element type is carried by the array descriptor/API call, matching
/// RPython's typed `ArrayDescr`; the runtime block itself stores only the
/// length prefix followed by inline bytes.
#[repr(C)]
pub struct GcTypedArray {
    /// Length prefix. Backend `bh_arraylen_gc` reads this word at offset 0.
    pub len: usize,
    /// Flexible-array tail marker. Actual items follow immediately in
    /// memory and are sized by the allocation descriptor.
    items: [u8; 0],
}

/// Array element kind — resume.py:656 arraydescr.is_array_of_* / FLAG_STRUCT parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayKind {
    Ref,
    Int,
    Float,
    /// Array(Struct(...)) — interior fields, item_size from arraydescr.
    Struct,
}

/// resume.py allocate_array, llmodel.py bh_new_array — API alias.
/// RPython: `bh_new_array_clear = bh_new_array` (llmodel.py).
/// Upstream both call `gc_malloc_array` which allocates a zero-filled
/// varsize block. Ref/int/float slots are word-sized.
pub fn allocate_array(length: usize, kind: ArrayKind, _clear: bool) -> *mut GcTypedArray {
    let item_size = match kind {
        ArrayKind::Ref | ArrayKind::Int | ArrayKind::Float => std::mem::size_of::<usize>(),
        ArrayKind::Struct => 0,
    };
    allocate_array_with_item_size(length, kind, item_size, _clear)
}

/// Same allocator as [`allocate_array`], but preserves the descriptor's
/// item size for array-of-structs materialization.
pub fn allocate_array_with_item_size(
    length: usize,
    _kind: ArrayKind,
    item_size: usize,
    _clear: bool,
) -> *mut GcTypedArray {
    allocate_flat_gc_typed_array(length, item_size)
}

/// resume.py VArrayStructInfo.allocate — API alias.
/// Allocate a flat byte buffer for Array(Struct(...)):
/// `[len][num_elems * item_size bytes]`.
pub fn allocate_array_struct(num_elems: usize, item_size: usize) -> *mut GcTypedArray {
    allocate_array_struct_at(num_elems, item_size, GC_TYPED_ARRAY_ITEMS_OFFSET)
}

/// Same block as [`allocate_array_struct`], with items at `items_base`.
///
/// `arraydescr.basesize` (`symbolic.get_array_token`) is the first item.
/// `GcEntries<i64, *mut PyObject>` aligns `[Entry; 0]` to 8, so a 4-byte
/// length word on wasm32 puts items at 8, not at [`GC_TYPED_ARRAY_ITEMS_OFFSET`].
/// The allocation is 8-aligned so that offset stays aligned for `u64`.
pub fn allocate_array_struct_at(
    num_elems: usize,
    item_size: usize,
    items_base: usize,
) -> *mut GcTypedArray {
    let items_size = num_elems
        .checked_mul(item_size)
        .expect("GcTypedArray item bytes overflow");
    let total = items_base
        .checked_add(items_size)
        .expect("GcTypedArray allocation size overflow");
    let align = std::mem::align_of::<u64>().max(std::mem::align_of::<GcTypedArray>());
    let layout = Layout::from_size_align(total, align).expect("GcTypedArray layout");
    unsafe {
        let raw = alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let array = raw.cast::<GcTypedArray>();
        (*array).len = num_elems;
        array
    }
}

/// Same payload as [`allocate_array_struct_at`], on the collector heap when
/// a collector is installed. `tid` is `ArrayDescr.tid`
/// (`gc.py` `init_array_descr` / `TypeLayoutBuilder.get_type_id`).
///
/// Header-less malloc is only the no-collector path (unit tests /
/// wasm-test). An unresolved tid with a collector installed is a bug:
/// the items would never be traced.
pub fn allocate_array_struct_at_typed(
    num_elems: usize,
    item_size: usize,
    items_base: usize,
    tid: u32,
) -> *mut GcTypedArray {
    if !majit_gc::gc_allocator_installed() {
        return allocate_array_struct_at(num_elems, item_size, items_base);
    }
    majit_ir::descr::assert_array_tid_for_malloc(tid, "allocate_array_struct_at_typed");
    let items_size = num_elems
        .checked_mul(item_size)
        .expect("GcTypedArray item bytes overflow");
    let total = items_base
        .checked_add(items_size)
        .expect("GcTypedArray allocation size overflow");
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(tid, total);
    assert!(
        !raw.is_null(),
        "allocate_array_struct_at_typed: collector installed but allocation had no route"
    );
    // `try_gc_alloc_stable_raw` is uninitialized. A virtual
    // `DICTENTRYARRAY` rebuild is `bh_new_array_clear`; leave
    // `f_valid` and the value word 0 like
    // `materialize_cleared_struct_gcarray`.
    unsafe {
        std::ptr::write_bytes(raw, 0, total);
        let array = raw.cast::<GcTypedArray>();
        (*array).len = num_elems;
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        array
    }
}

fn allocate_flat_gc_typed_array(length: usize, item_size: usize) -> *mut GcTypedArray {
    let layout = gc_typed_array_layout(length, item_size);
    unsafe {
        let raw = alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let array = raw as *mut GcTypedArray;
        (*array).len = length;
        array
    }
}

fn gc_typed_array_layout(length: usize, item_size: usize) -> Layout {
    let items_size = length
        .checked_mul(item_size)
        .expect("GcTypedArray item bytes overflow");
    let total = GC_TYPED_ARRAY_ITEMS_OFFSET
        .checked_add(items_size)
        .expect("GcTypedArray allocation size overflow");
    Layout::from_size_align(total, std::mem::align_of::<GcTypedArray>())
        .expect("GcTypedArray layout")
}

/// Return the items base pointer of a `GcTypedArray`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn gc_typed_array_items_base(array: *mut GcTypedArray) -> *mut u8 {
    if array.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (array as *mut u8).add(GC_TYPED_ARRAY_ITEMS_OFFSET) }
}

/// llmodel.py bh_getarrayitem_gc_r bh_get/setarrayitem_gc_* access the raw byte
/// offset with no bounds check — a null array or out-of-range index
/// never occurs in correct jitcode. Panic loudly instead of reading
/// or writing out of bounds.
#[inline]
unsafe fn gc_typed_array_item_ptr(
    array: *mut GcTypedArray,
    index: usize,
    item_size: usize,
) -> *mut u8 {
    assert!(!array.is_null(), "gc_typed_array_item_ptr: null array");
    let len = unsafe { (*array).len };
    assert!(
        index < len,
        "gc_typed_array_item_ptr: index {index} out of range (len {len})"
    );
    unsafe { gc_typed_array_items_base(array).add(index * item_size) }
}

/// llmodel.py bh_setarrayitem_gc_r parity.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn setarrayitem_ref(array: *mut GcTypedArray, index: usize, value: PyObjectRef) {
    unsafe {
        let ptr = gc_typed_array_item_ptr(array, index, std::mem::size_of::<PyObjectRef>());
        (ptr as *mut PyObjectRef).write_unaligned(value);
    }
}

/// llmodel.py bh_getarrayitem_gc_r parity.
/// Read a `PyObjectRef` from a ref array slot.
pub fn getarrayitem_ref(array: *const GcTypedArray, index: usize) -> PyObjectRef {
    unsafe {
        let ptr = gc_typed_array_item_ptr(
            array as *mut GcTypedArray,
            index,
            std::mem::size_of::<PyObjectRef>(),
        );
        (ptr as *const PyObjectRef).read_unaligned()
    }
}

/// llmodel.py bh_setarrayitem_gc_i parity.
/// Write a raw i64 to an int array slot.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn setarrayitem_int(array: *mut GcTypedArray, index: usize, value: i64) {
    unsafe {
        let ptr = gc_typed_array_item_ptr(array, index, std::mem::size_of::<i64>());
        (ptr as *mut i64).write_unaligned(value);
    }
}

/// llmodel.py bh_setarrayitem_gc_f parity.
/// Write a raw f64 to a float array slot.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn setarrayitem_float(array: *mut GcTypedArray, index: usize, value: f64) {
    unsafe {
        let ptr = gc_typed_array_item_ptr(array, index, std::mem::size_of::<f64>());
        (ptr as *mut f64).write_unaligned(value);
    }
}

/// resume.py setinteriorfield(i, array, num, fielddescrs[j]) parity.
/// resume.py ResumeDataDirectReader: dispatch on descr type.
/// llmodel.py bh_setinteriorfield_gc_i: byte offset =
/// `basesize + elem_idx * item_size + field_offset`.
///
/// `items_base` is that `basesize`. A flat [`GcTypedArray`] passes
/// [`GC_TYPED_ARRAY_ITEMS_OFFSET`]. A length-prefixed `GcArray<T>` whose
/// element is aligned past the length word passes the wider offset.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn setinteriorfield(
    array: *mut GcTypedArray,
    items_base: usize,
    elem_idx: usize,
    field_offset: usize,
    field_size: usize,
    item_size: usize,
    descr_field_type: u8,
    value: i64,
) {
    if array.is_null() {
        return;
    }
    if item_size == 0 || field_size == 0 {
        return;
    }
    let len = gcarray_len(array);
    let Some(byte_offset) = elem_idx
        .checked_mul(item_size)
        .and_then(|base| base.checked_add(field_offset))
    else {
        return;
    };
    let Some(end) = byte_offset.checked_add(field_size) else {
        return;
    };
    let Some(total) = len.checked_mul(item_size) else {
        return;
    };
    if end > total {
        return;
    }
    let Some(abs) = items_base.checked_add(byte_offset) else {
        return;
    };
    unsafe {
        let ptr = (array as *mut u8).add(abs);
        match descr_field_type {
            2 => {
                let bits = value as u64;
                std::ptr::copy_nonoverlapping(bits.to_ne_bytes().as_ptr(), ptr, field_size.min(8));
            }
            0 => {
                let raw = value as usize;
                std::ptr::copy_nonoverlapping(
                    raw.to_ne_bytes().as_ptr(),
                    ptr,
                    field_size.min(std::mem::size_of::<usize>()),
                );
            }
            _ => {
                std::ptr::copy_nonoverlapping(value.to_ne_bytes().as_ptr(), ptr, field_size.min(8));
            }
        }
    }
}

/// Resume parity: get the length of a GcTypedArray.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn gcarray_len(array: *const GcTypedArray) -> usize {
    if array.is_null() {
        return 0;
    }
    unsafe { (*array).len }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `std_gc_array_size` accepts exactly the sizes `Layout::from_size_align`
    /// accepts at the header's alignment, so `std_gc_array_layout` never
    /// reaches its `expect` for an accepted size.
    #[test]
    fn rlist_overallocate_of_nine_is_sixteen() {
        // rlist.py: newsize=9, some=6+(9>>3)=7, new_allocated=16.
        assert_eq!(rlist_overallocate_signed(9), 16);
        assert_eq!(rlist_overallocate_cap(9), 16);
        assert_eq!(rlist_overallocate_signed(8), 8 + 3 + 1);
    }

    #[test]
    fn std_gc_array_size_matches_layout_bound() {
        let payload = Layout::from_size_align(24, 8).unwrap();
        assert_eq!(std_gc_array_size(payload), Some(GC_ARRAY_HEADER_SIZE + 24));
        assert_eq!(
            std_gc_array_layout(GC_ARRAY_HEADER_SIZE + 24),
            Layout::from_size_align(GC_ARRAY_HEADER_SIZE + 24, GC_ARRAY_HEADER_ALIGN).unwrap()
        );
        let bound = isize::MAX as usize - (GC_ARRAY_HEADER_ALIGN - 1);
        let fits = Layout::from_size_align(bound - GC_ARRAY_HEADER_SIZE, 8).unwrap();
        assert_eq!(std_gc_array_size(fits), Some(bound));
        assert!(Layout::from_size_align(bound, GC_ARRAY_HEADER_ALIGN).is_ok());
        let over = Layout::from_size_align(bound - GC_ARRAY_HEADER_SIZE + 1, 1).unwrap();
        assert_eq!(std_gc_array_size(over), None);
        assert!(Layout::from_size_align(bound + 1, GC_ARRAY_HEADER_ALIGN).is_err());
    }

    #[test]
    fn allocate_array_struct_at_typed_falls_back_without_a_collector() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let arr = allocate_array_struct_at_typed(2, 16, GC_TYPED_ARRAY_ITEMS_OFFSET, 0);
        assert!(!arr.is_null());
        assert_eq!(gcarray_len(arr), 2);
        let arr = allocate_array_struct_at_typed(
            1,
            16,
            GC_TYPED_ARRAY_ITEMS_OFFSET,
            majit_rlib::lltypesystem::rlist::UNSET_GC_TYPE_ID,
        );
        assert!(!arr.is_null());
        assert_eq!(gcarray_len(arr), 1);
    }

    #[test]
    fn gc_typed_array_ref_roundtrip() {
        let arr = allocate_array(3, ArrayKind::Ref, true);
        assert_eq!(gcarray_len(arr), 3);
        // Zero-filled allocation: every slot reads back null.
        for i in 0..3 {
            assert!(getarrayitem_ref(arr, i).is_null());
        }
        let a = 0x1000usize as PyObjectRef;
        let b = 0x2008usize as PyObjectRef;
        setarrayitem_ref(arr, 0, a);
        setarrayitem_ref(arr, 2, b);
        assert_eq!(getarrayitem_ref(arr, 0), a);
        assert!(getarrayitem_ref(arr, 1).is_null());
        assert_eq!(getarrayitem_ref(arr, 2), b);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn gc_typed_array_out_of_range_read_panics() {
        let arr = allocate_array(3, ArrayKind::Ref, true);
        getarrayitem_ref(arr, 3);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn gc_typed_array_out_of_range_write_panics() {
        let arr = allocate_array(3, ArrayKind::Ref, true);
        setarrayitem_ref(arr, 3, std::ptr::null_mut());
    }

    #[test]
    fn jit_ll_arraymove_preserves_overlap_in_both_directions() {
        let values = || {
            [
                1usize as PyObjectRef,
                2usize as PyObjectRef,
                3usize as PyObjectRef,
                4usize as PyObjectRef,
            ]
        };
        let left = unsafe { alloc_list_items_block(&values()) };
        jit_ll_arraymove(left as crate::PyObjectRef, 1, 0, 3);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(items_block_items_base(left), 4) },
            &[
                2usize as PyObjectRef,
                3usize as PyObjectRef,
                4usize as PyObjectRef,
                4usize as PyObjectRef,
            ]
        );
        unsafe { dealloc_list_items_block(left) };

        let right = unsafe { alloc_list_items_block(&values()) };
        jit_ll_arraymove(right as crate::PyObjectRef, 0, 1, 3);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(items_block_items_base(right), 4) },
            &[
                1usize as PyObjectRef,
                1usize as PyObjectRef,
                2usize as PyObjectRef,
                3usize as PyObjectRef,
            ]
        );
        unsafe { dealloc_list_items_block(right) };

        // `rgc.ll_arraymove` keeps this case as an explicit `copy_item`
        // before entering the barrier-plus-memmove body.
        let one = unsafe { alloc_list_items_block(&values()) };
        jit_ll_arraymove(one as crate::PyObjectRef, 3, 0, 1);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(items_block_items_base(one), 4) },
            &[
                4usize as PyObjectRef,
                2usize as PyObjectRef,
                3usize as PyObjectRef,
                4usize as PyObjectRef,
            ]
        );
        unsafe { dealloc_list_items_block(one) };
    }

    #[test]
    fn jit_ll_arraycopy_copies_a_range_between_blocks() {
        let src = unsafe {
            alloc_list_items_block(&[
                1usize as PyObjectRef,
                2usize as PyObjectRef,
                3usize as PyObjectRef,
                4usize as PyObjectRef,
            ])
        };
        let dst = unsafe {
            alloc_list_items_block(&[
                0usize as PyObjectRef,
                0usize as PyObjectRef,
                0usize as PyObjectRef,
                0usize as PyObjectRef,
            ])
        };
        jit_ll_arraycopy(
            src as crate::PyObjectRef,
            dst as crate::PyObjectRef,
            1,
            0,
            3,
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(items_block_items_base(dst), 4) },
            &[
                2usize as PyObjectRef,
                3usize as PyObjectRef,
                4usize as PyObjectRef,
                0usize as PyObjectRef,
            ]
        );
        unsafe {
            dealloc_list_items_block(src);
            dealloc_list_items_block(dst);
        }
    }

    #[test]
    fn jit_ll_arraycopy_length_one_is_copy_item() {
        let src =
            unsafe { alloc_list_items_block(&[1usize as PyObjectRef, 2usize as PyObjectRef]) };
        let dst =
            unsafe { alloc_list_items_block(&[0usize as PyObjectRef, 0usize as PyObjectRef]) };
        jit_ll_arraycopy(
            src as crate::PyObjectRef,
            dst as crate::PyObjectRef,
            1,
            0,
            1,
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(items_block_items_base(dst), 2) },
            &[2usize as PyObjectRef, 0usize as PyObjectRef]
        );
        unsafe {
            dealloc_list_items_block(src);
            dealloc_list_items_block(dst);
        }
    }

    /// The store address is the caller's items base, not the flat header.
    /// A 64-bit host gives both the same offset, so the base here sits
    /// 16 bytes past [`GC_TYPED_ARRAY_ITEMS_OFFSET`].
    #[test]
    fn setinteriorfield_writes_at_the_descr_items_base() {
        let items_base = GC_TYPED_ARRAY_ITEMS_OFFSET + 16;
        let item_size = 8usize;
        let array = allocate_array_struct_at(2, item_size, items_base);
        setinteriorfield(array, items_base, 1, 0, 1, item_size, 1, 0x5a);
        unsafe {
            let raw = array.cast::<u8>();
            assert_eq!(raw.add(GC_TYPED_ARRAY_ITEMS_OFFSET).read(), 0);
            assert_eq!(raw.add(items_base + item_size).read(), 0x5a);
            let total = items_base + 2 * item_size;
            let align = std::mem::align_of::<u64>().max(std::mem::align_of::<GcTypedArray>());
            std::alloc::dealloc(
                raw,
                std::alloc::Layout::from_size_align(total, align).unwrap(),
            );
        }
    }
}
