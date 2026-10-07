//! `rpython/rtyper/lltypesystem/rlist.py` for a Rust `alloc::vec::Vec<T>`.
//!
//! The translator lowers a `Vec<T>` to a pointer to its raw three-word header
//! (`Ptr(Struct(raw) {ptr, len, cap})`, word indices in [`majit_ir::rvec`])
//! and its methods to the `ll_vec_*` helpers here, as `rtyper/rlist.py` lowers
//! an RPython list to the `ll_*` helpers of `lltypesystem/rlist.py`.
//!
//! Items are one word each. A helper exists per item register kind — `_i`
//! (`usize`), `_r` (`*mut u8`, a managed reference) and `_f` (`f64`) — because
//! the translator reads bodies before monomorphisation. The hints are those of
//! the upstream helper each one mirrors:
//!
//! * `ll_vec_length_*`, `ll_vec_getitem_fast_*`, `ll_vec_setitem_fast_*`,
//!   `ll_vec_newlist_hint_*` and `ll_vec_newemptylist_*` carry the upstream
//!   oopspecs (`list.len`, `list.getitem`, `list.setitem`, `newlist_hint`,
//!   `newlist`), which jtransform rewrites into raw header and item accesses.
//! * `ll_vec_append_*` and `ll_vec_resize_ge_*` are looked inside;
//!   `ll_vec_resize_hint_really_*` and `ll_vec_reverse_*` are looked inside
//!   only under their upstream `look_inside_iff` predicates.
//! * Only the buffer allocation itself is opaque: [`vec_buf_alloc`],
//!   [`vec_buf_realloc`] and [`vec_buf_free`] use the same `Layout` as the
//!   standard library's `RawVec`, so a buffer grown here can be freed by host
//!   code and the reverse.
//!
//! The item buffer stays raw memory for every kind, managed references
//! included: the interpreter treats `Vec` contents as untraced and roots them
//! explicitly where it allocates, and the lowered code keeps that contract.
//!
//! `ll_vec_free_*` frees a header the lowering allocated with
//! [`raw_malloc_varsize_char`] together with its buffer; only lowered code
//! reaches it.
//!
//! A borrowed slice of the same items is the Rust fat pointer, the two values
//! `(items, length)`. `ll_vec_items_*` reads a header's item pointer, the
//! first of them (`ll_items`); the length is `ll_vec_length_*`. The
//! `ll_slice_*` helpers take the item pointer as an address, and the length
//! where they need it, as separate word arguments.

use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, handle_alloc_error, realloc};

// The header word indices, spelled in this crate so a helper body reads a
// constant it defines; the assertions keep them equal to the layout
// `majit_ir::rvec` describes.
const VEC_CAP_WORD: usize = 0;
const VEC_PTR_WORD: usize = 1;
const VEC_LEN_WORD: usize = 2;
const _: () = assert!(VEC_CAP_WORD == majit_ir::rvec::VEC_CAP_WORD);
const _: () = assert!(VEC_PTR_WORD == majit_ir::rvec::VEC_PTR_WORD);
const _: () = assert!(VEC_LEN_WORD == majit_ir::rvec::VEC_LEN_WORD);

/// In-place reverse of `items[start..end]`, the swap loop of `rlist.ll_reverse`.
/// Expanded at each `ll_slice_rotate_*` site.
macro_rules! ll_slice_reverse_range {
    ($getitem:ident, $setitem:ident, $items:ident, $start:expr, $end:expr) => {{
        let mut i = $start as isize;
        let mut j = $end as isize - 1;
        while i < j {
            let tmp = $getitem($items, i as usize);
            let other = $getitem($items, j as usize);
            $setitem($items, i as usize, other);
            $setitem($items, j as usize, tmp);
            i += 1;
            j -= 1;
        }
    }};
}

/// In-place reverse of a managed-reference range; items move as address words,
/// the same as [`ll_slice_reverse_r`].
macro_rules! ll_slice_reverse_range_r {
    ($items:ident, $start:expr, $end:expr) => {{
        let mut i = $start as isize;
        let mut j = $end as isize - 1;
        while i < j {
            let low = slice_item_addr($items, i as usize, ITEM_SIZE_R);
            let high = slice_item_addr($items, j as usize, ITEM_SIZE_R);
            let tmp = raw_read_ptr(low);
            raw_write_ptr(low, raw_read_ptr(high));
            raw_write_ptr(high, tmp);
            i += 1;
            j -= 1;
        }
    }};
}

/// Three-reverse rotate of a pair slice. `$right` is `rotate_right`.
macro_rules! ll_slice_rotate_body {
    ($items:ident, $length:ident, $k:ident, $reverse_range:ident, $right:expr) => {{
        if $length <= 1 {
            return;
        }
        let k = if $k >= $length { $k % $length } else { $k };
        if k == 0 {
            return;
        }
        if $right {
            $reverse_range!($items, 0, $length);
            $reverse_range!($items, 0, k);
            $reverse_range!($items, k, $length);
        } else {
            $reverse_range!($items, 0, k);
            $reverse_range!($items, k, $length);
            $reverse_range!($items, 0, $length);
        }
    }};
}

macro_rules! ll_slice_reverse_range_i {
    ($items:ident, $start:expr, $end:expr) => {
        ll_slice_reverse_range!(
            ll_slice_getitem_fast_i,
            ll_slice_setitem_fast_i,
            $items,
            $start,
            $end
        )
    };
}

macro_rules! ll_slice_reverse_range_f {
    ($items:ident, $start:expr, $end:expr) => {
        ll_slice_reverse_range!(
            ll_slice_getitem_fast_f,
            ll_slice_setitem_fast_f,
            $items,
            $start,
            $end
        )
    };
}

use super::rffi::{
    raw_free, raw_malloc_varsize_char, raw_ptradd, raw_read_f64, raw_read_ptr, raw_write_f64,
    raw_write_ptr,
};

const WORD: usize = std::mem::size_of::<usize>();

/// Standard GcArray length word (`majit_gc` `standard_array_length_ofs`).
const GCARRAY_LEN_OFFSET: usize = 0;
/// Word-sized items follow the length word (`array_items_base` for a
/// pointer element: the length word rounded up to the item's alignment).
const GCARRAY_ITEMS_OFFSET: usize = WORD;

const ITEM_SIZE_I: usize = std::mem::size_of::<usize>();
const ITEM_ALIGN_I: usize = std::mem::align_of::<usize>();
const ITEM_SIZE_R: usize = std::mem::size_of::<*mut u8>();
const ITEM_ALIGN_R: usize = std::mem::align_of::<*mut u8>();
const ITEM_SIZE_F: usize = std::mem::size_of::<f64>();
const ITEM_ALIGN_F: usize = std::mem::align_of::<f64>();

// ── header words ────────────────────────────────────────────────────────

fn vec_header_word(header: usize, index: usize) -> usize {
    raw_read_ptr(raw_ptradd(header, index * WORD))
}

fn vec_set_header_word(header: usize, index: usize, value: usize) {
    raw_write_ptr(raw_ptradd(header, index * WORD), value)
}

fn vec_item_addr(header: usize, index: usize, itemsize: usize) -> usize {
    raw_ptradd(vec_header_word(header, VEC_PTR_WORD), index * itemsize)
}

fn vec_header_i(l: &mut Vec<usize>) -> usize {
    l as *mut Vec<usize> as usize
}

/// Buffer pointer word of any `Vec<T>` header (`rlist.py` `ll_items`:
/// `return l.items`). Item-kind independent: every `Vec<T>` header stores
/// that word at `VEC_PTR_WORD`, including items the one-word `ll_vec_items_*`
/// helpers do not name.
pub fn ll_vec_as_ptr(header: usize) -> usize {
    vec_header_word(header, VEC_PTR_WORD)
}

fn vec_header_r(l: &mut Vec<*mut u8>) -> usize {
    l as *mut Vec<*mut u8> as usize
}

fn vec_header_f(l: &mut Vec<f64>) -> usize {
    l as *mut Vec<f64> as usize
}

/// `_ll_list_resize_hint_really`'s capacity choice for a positive `newsize`.
fn vec_new_allocated(newsize: usize, overallocate: bool) -> usize {
    if overallocate {
        let mut some = if newsize < 9 { 3 } else { 6 };
        some += newsize >> 3;
        newsize + some
    } else {
        newsize
    }
}

/// Address of item `index` of the items at `items`.
fn slice_item_addr(items: usize, index: usize, itemsize: usize) -> usize {
    raw_ptradd(items, index * itemsize)
}

// ── slice length out-param ──────────────────────────────────────────────

/// `lltype.scoped_alloc(rffi.SIGNEDP.TO, 1)`: the one-word raw slot a caller
/// passes, as an extra trailing argument, to a function that returns a slice.
/// The callee stores the slice length into it and returns the item pointer.
pub fn ll_slice_len_slot_new() -> usize {
    raw_malloc_varsize_char(WORD)
}

/// The callee side: store the returned slice's length into the caller's slot.
pub fn ll_slice_len_slot_store(slot: usize, length: usize) {
    raw_write_ptr(slot, length)
}

/// The caller side: read the returned slice's length and free the slot at
/// the end of its scope, which is this read.
pub fn ll_slice_len_slot_take(slot: usize) -> usize {
    let length = raw_read_ptr(slot);
    raw_free(slot);
    length
}

// ── raw item buffer of an array borrowed as a slice ─────────────────────

/// `lltype.free(buf, flavor='raw')` of an `ll_slice_buffer_new_*` buffer when
/// the array's storage ends.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_free(items: usize) {
    raw_free(items)
}

// ── opaque buffer allocation ────────────────────────────────────────────

fn vec_buf_layout(allocated: usize, itemsize: usize, align: usize) -> Layout {
    item_bytes(allocated, itemsize)
        .and_then(|size| Layout::from_size_align(size, align).ok())
        .unwrap_or_else(|| panic!("Vec capacity overflow"))
}

fn item_bytes(count: usize, itemsize: usize) -> Option<usize> {
    count.checked_mul(itemsize)
}

/// A buffer for `allocated` items. An empty buffer is the dangling address
/// `align`, as `RawVec` spells it.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn vec_buf_alloc(allocated: usize, itemsize: usize, align: usize) -> usize {
    if allocated == 0 || itemsize == 0 {
        return align;
    }
    let layout = vec_buf_layout(allocated, itemsize, align);
    let items = unsafe { alloc(layout) };
    if items.is_null() {
        handle_alloc_error(layout);
    }
    items as usize
}

/// A zero-filled buffer for `allocated` items (`raw_malloc(..., zero=True)`).
/// An empty buffer is the dangling address `align`, as [`vec_buf_alloc`].
#[majit_macros::dont_look_inside_cannot_raise]
pub fn vec_buf_alloc_clear(allocated: usize, itemsize: usize, align: usize) -> usize {
    if allocated == 0 || itemsize == 0 {
        return align;
    }
    let layout = vec_buf_layout(allocated, itemsize, align);
    let items = unsafe { alloc_zeroed(layout) };
    if items.is_null() {
        handle_alloc_error(layout);
    }
    items as usize
}

/// Move `items` (holding `allocated` slots) to a buffer of `new_allocated`
/// slots, keeping the first `min(allocated, new_allocated)` items.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn vec_buf_realloc(
    items: usize,
    allocated: usize,
    new_allocated: usize,
    itemsize: usize,
    align: usize,
) -> usize {
    if allocated == 0 || itemsize == 0 {
        return vec_buf_alloc(new_allocated, itemsize, align);
    }
    if new_allocated == 0 {
        vec_buf_free(items, allocated, itemsize, align);
        return align;
    }
    let old_layout = vec_buf_layout(allocated, itemsize, align);
    let new_layout = vec_buf_layout(new_allocated, itemsize, align);
    let newitems = unsafe { realloc(items as *mut u8, old_layout, new_layout.size()) };
    if newitems.is_null() {
        handle_alloc_error(new_layout);
    }
    newitems as usize
}

/// Free a buffer of `allocated` slots.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn vec_buf_free(items: usize, allocated: usize, itemsize: usize, align: usize) {
    if allocated == 0 || itemsize == 0 {
        return;
    }
    unsafe { dealloc(items as *mut u8, vec_buf_layout(allocated, itemsize, align)) }
}

// ── `_i`: `usize` items ─────────────────────────────────────────────────

/// `ll_newemptylist`.
#[majit_macros::oopspec("newlist(0)")]
pub fn ll_vec_newemptylist_i() -> Vec<usize> {
    Vec::new()
}

/// `ll_newlist_hint`.
#[majit_macros::oopspec("newlist_hint(lengthhint)")]
pub fn ll_vec_newlist_hint_i(lengthhint: usize) -> Vec<usize> {
    Vec::with_capacity(lengthhint)
}

/// `lltypesystem/rlist.py ll_newlist`: `length` slots. Each slot is zero
/// until the caller writes it, and `length` is already the list length.
/// No `newlist(length)` oopspec: the result is a raw `Vec` header (kind
/// int), and that rewrite emits GC `new_array_clear` into a ref bank.
pub fn ll_vec_newlist_i(length: usize) -> Vec<usize> {
    let mut l = Vec::with_capacity(length);
    unsafe {
        l.set_len(length);
    }
    ll_vec_arrayclear_i(&mut l, length);
    l
}

/// `rlist.py _ll_zero_or_null` for a word item.
fn ll_vec_zero_or_null_i(item: usize) -> bool {
    item == 0
}

/// `rgc.ll_arrayclear`. Writes zero into each of the `count` slots.
#[majit_macros::dont_look_inside_cannot_raise]
fn ll_vec_arrayclear_i(l: &mut Vec<usize>, count: usize) {
    let mut i = 0;
    while i < count {
        ll_vec_setitem_fast_i(l, i, 0);
        i += 1;
    }
}

/// `rlist.py _ll_alloc_and_clear`.
/// No `newlist_clear` oopspec: this is a raw `Vec`, not a GC list header.
pub fn ll_vec_alloc_and_clear_i(count: usize) -> Vec<usize> {
    let mut l = ll_vec_newlist_i(count);
    ll_vec_arrayclear_i(&mut l, count);
    l
}

/// `@jit.look_inside_iff(lambda LIST, count, item: jit.isconstant(count) and count < 137)`.
fn ll_vec_alloc_and_set_nonnull_iff_i(count: usize, _item: usize) -> bool {
    crate::jit::isconstant(&count) && count < 137
}

/// `rlist.py _ll_alloc_and_set_nonnull`.
#[majit_macros::look_inside_iff(ll_vec_alloc_and_set_nonnull_iff_i)]
pub fn ll_vec_alloc_and_set_nonnull_i(count: usize, item: usize) -> Vec<usize> {
    let mut l = ll_vec_newlist_i(count);
    let mut i = 0;
    while i < count {
        ll_vec_setitem_fast_i(&mut l, i, item);
        i += 1;
    }
    l
}

/// `rlist.py _ll_alloc_and_set_nojit`. The non-zero arm is `rgc.ll_arrayfill`
/// (`ll_slice_buffer_fill_i`).
fn ll_vec_alloc_and_set_nojit_i(count: usize, item: usize) -> Vec<usize> {
    let mut l = ll_vec_newlist_i(count);
    if ll_vec_zero_or_null_i(item) {
        ll_vec_arrayclear_i(&mut l, count);
    } else {
        ll_slice_buffer_fill_i(ll_vec_items_i(&mut l), count, item);
    }
    l
}

/// `rlist.py _ll_alloc_and_set_jit`.
fn ll_vec_alloc_and_set_jit_i(count: usize, item: usize) -> Vec<usize> {
    if ll_vec_zero_or_null_i(item) {
        ll_vec_alloc_and_clear_i(count)
    } else {
        ll_vec_alloc_and_set_nonnull_i(count, item)
    }
}

/// `rlist.py ll_alloc_and_set`. `rarithmetic.int_force_ge_zero` is a no-op:
/// `count` is `usize`, already `>= 0`.
pub fn ll_vec_alloc_and_set_i(count: usize, item: usize) -> Vec<usize> {
    if crate::jit::we_are_jitted() {
        ll_vec_alloc_and_set_jit_i(count, item)
    } else {
        ll_vec_alloc_and_set_nojit_i(count, item)
    }
}

/// `ll_length`.
#[majit_macros::oopspec("list.len(l)")]
pub fn ll_vec_length_i(l: &mut Vec<usize>) -> usize {
    vec_header_word(vec_header_i(l), VEC_LEN_WORD)
}

/// `ll_getitem_fast`: `index` is in bounds.
#[majit_macros::oopspec("list.getitem(l, index)")]
pub fn ll_vec_getitem_fast_i(l: &mut Vec<usize>, index: usize) -> usize {
    raw_read_ptr(vec_item_addr(vec_header_i(l), index, ITEM_SIZE_I))
}

/// `ll_setitem_fast`: `index` is in bounds.
#[majit_macros::oopspec("list.setitem(l, index, item)")]
pub fn ll_vec_setitem_fast_i(l: &mut Vec<usize>, index: usize, item: usize) {
    raw_write_ptr(vec_item_addr(vec_header_i(l), index, ITEM_SIZE_I), item)
}

/// `ll_append`.
pub fn ll_vec_append_i(l: &mut Vec<usize>, newitem: usize) {
    let length = ll_vec_length_i(l);
    ll_vec_resize_ge_i(l, length + 1);
    ll_vec_setitem_fast_i(l, length, newitem);
}

/// `_ll_list_resize_ge`.
pub fn ll_vec_resize_ge_i(l: &mut Vec<usize>, newsize: usize) {
    let allocated = vec_header_word(vec_header_i(l), VEC_CAP_WORD);
    let cond = allocated < newsize;
    if crate::jit::isconstant(&allocated) && crate::jit::isconstant(&newsize) {
        if cond {
            ll_vec_resize_hint_really_i(l, newsize, true);
        }
    } else {
        crate::jit::conditional_call3(cond, ll_vec_resize_hint_really_i, &mut *l, newsize, true);
    }
    vec_set_header_word(vec_header_i(l), VEC_LEN_WORD, newsize);
}

/// `@jit.look_inside_iff(lambda l, newsize, overallocate:
/// jit.isconstant(len(l.items)) and jit.isconstant(newsize))`.
fn ll_vec_resize_hint_really_iff_i(
    l: &mut Vec<usize>,
    newsize: usize,
    _overallocate: bool,
) -> bool {
    let allocated = vec_header_word(vec_header_i(l), VEC_CAP_WORD);
    crate::jit::isconstant(&allocated) && crate::jit::isconstant(&newsize)
}

/// `_ll_list_resize_hint_really`.
#[majit_macros::look_inside_iff(ll_vec_resize_hint_really_iff_i)]
pub fn ll_vec_resize_hint_really_i(l: &mut Vec<usize>, newsize: usize, overallocate: bool) {
    let header = vec_header_i(l);
    let new_allocated = if newsize == 0 {
        vec_set_header_word(header, VEC_LEN_WORD, 0);
        0
    } else {
        vec_new_allocated(newsize, overallocate)
    };
    let items = vec_header_word(header, VEC_PTR_WORD);
    let allocated = vec_header_word(header, VEC_CAP_WORD);
    let newitems = vec_buf_realloc(items, allocated, new_allocated, ITEM_SIZE_I, ITEM_ALIGN_I);
    vec_set_header_word(header, VEC_PTR_WORD, newitems);
    vec_set_header_word(header, VEC_CAP_WORD, new_allocated);
}

/// `@jit.look_inside_iff(lambda l: jit.isvirtual(l) and
/// jit.isconstant(l.ll_length()))`.
fn ll_vec_reverse_iff_i(l: &mut Vec<usize>) -> bool {
    let length = ll_vec_length_i(l);
    crate::jit::isvirtual(&*l) && crate::jit::isconstant(&length)
}

/// `ll_reverse`.
#[majit_macros::look_inside_iff(ll_vec_reverse_iff_i)]
pub fn ll_vec_reverse_i(l: &mut Vec<usize>) {
    let length = ll_vec_length_i(l) as isize;
    let mut i: isize = 0;
    let mut length_1_i = length - 1 - i;
    while i < length_1_i {
        let tmp = ll_vec_getitem_fast_i(l, i as usize);
        let other = ll_vec_getitem_fast_i(l, length_1_i as usize);
        ll_vec_setitem_fast_i(l, i as usize, other);
        ll_vec_setitem_fast_i(l, length_1_i as usize, tmp);
        i += 1;
        length_1_i -= 1;
    }
}

/// `lltype.free(l, flavor='raw')` of a lowered `Vec` header and its buffer.
pub fn ll_vec_free_i(l: &mut Vec<usize>) {
    let header = vec_header_i(l);
    let items = vec_header_word(header, VEC_PTR_WORD);
    let allocated = vec_header_word(header, VEC_CAP_WORD);
    vec_buf_free(items, allocated, ITEM_SIZE_I, ITEM_ALIGN_I);
    raw_free(header);
}

/// `ll_items`: the header's item pointer.
pub fn ll_vec_items_i(l: &mut Vec<usize>) -> usize {
    vec_header_word(vec_header_i(l), VEC_PTR_WORD)
}

/// `ll_extend` from the `(items, length)` slice. `ovfcheck` on `len1 + len2`
/// rejects a wrapping sum, then `_ll_resize_ge` and `ll_arraycopy`.
pub fn ll_vec_extend_from_slice_i(l: &mut Vec<usize>, items: usize, length: usize) {
    let len1 = ll_vec_length_i(l);
    let newlen = len1.checked_add(length).expect("Vec capacity overflow");
    ll_vec_resize_ge_i(l, newlen);
    ll_slice_arraycopy_i(items, ll_vec_items_i(l), 0, len1, length);
}

/// Panic for a pair-slice index or `copy_from_slice` length mismatch.
#[majit_macros::dont_look_inside]
pub fn ll_slice_bounds_panic() {
    panic!("slice index out of bounds");
}

/// `rgc.ll_arraycopy` over raw items. The items hold no GC pointer, so the
/// copy is the `raw_memcopy` of `length` items; `ll_arraycopy` is a residual
/// call (`list.ll_arraycopy`, `OS_ARRAYCOPY`), and so is this.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_arraycopy_i(
    source: usize,
    dest: usize,
    source_start: usize,
    dest_start: usize,
    length: usize,
) {
    let nbytes = item_bytes(length, ITEM_SIZE_I).expect("Vec capacity overflow");
    unsafe {
        std::ptr::copy_nonoverlapping(
            slice_item_addr(source, source_start, ITEM_SIZE_I) as *const u8,
            slice_item_addr(dest, dest_start, ITEM_SIZE_I) as *mut u8,
            nbytes,
        );
    }
}

/// `ll_getitem_fast` on the items at `items`: `index` is in bounds.
pub fn ll_slice_getitem_fast_i(items: usize, index: usize) -> usize {
    raw_read_ptr(slice_item_addr(items, index, ITEM_SIZE_I))
}

/// `ll_setitem_fast` on the items at `items`: `index` is in bounds.
pub fn ll_slice_setitem_fast_i(items: usize, index: usize, item: usize) {
    raw_write_ptr(slice_item_addr(items, index, ITEM_SIZE_I), item)
}

/// `ll_listcontains` on the `(items, length)` slice.
///
/// `ll_listcontains` is "not inlined by the JIT -- contains a loop".
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_contains_i(items: usize, length: usize, item: usize) -> bool {
    let mut i = 0;
    while i < length {
        if ll_slice_getitem_fast_i(items, i) == item {
            return true;
        }
        i += 1;
    }
    false
}

/// The item at `addr`, an [`ll_slice_get_addr_i`] result, or `default` when
/// `addr` is 0: `opt.copied().unwrap_or(default)`.
pub fn ll_slice_load_or_i(addr: usize, default: usize) -> usize {
    if addr != 0 {
        ll_slice_getitem_fast_i(addr, 0)
    } else {
        default
    }
}

/// `s.to_vec()`: `ll_copy` — a new `Vec` of `length` items, then
/// `ll_arraycopy` of the items at `items`.
pub fn ll_slice_to_vec_i(items: usize, length: usize) -> Vec<usize> {
    let mut l = ll_vec_newlist_hint_i(length);
    ll_vec_resize_ge_i(&mut l, length);
    ll_slice_arraycopy_i(items, ll_vec_items_i(&mut l), 0, 0, length);
    l
}

/// The item pointer of `&s[start..]` over the items at `items`: `start` is
/// at most the slice's length.
pub fn ll_slice_offset_i(items: usize, start: usize) -> usize {
    slice_item_addr(items, start, ITEM_SIZE_I)
}

/// The address of item `index` of `(items, length)`, or 0 when `index` is
/// out of bounds: the one-word `Option<&T>` of `s.get(index)`.
pub fn ll_slice_get_addr_i(items: usize, length: usize, index: usize) -> usize {
    if index < length {
        slice_item_addr(items, index, ITEM_SIZE_I)
    } else {
        0
    }
}

/// `lltype.malloc(Array(ITEM), length, flavor='raw')`: the item buffer of an
/// array the lowering keeps in raw memory because it is borrowed as a slice.
/// Opaque like [`vec_buf_alloc`]. `checked_mul` overflow stays in this
/// residual; `raw_malloc_varsize_char` is already dont_look_inside.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_new_i(length: usize) -> usize {
    let size = item_bytes(length, ITEM_SIZE_I).expect("Vec capacity overflow");
    raw_malloc_varsize_char(size)
}

/// `[item; length]` into the buffer at `items`.
/// `rgc.ll_arrayfill`, which is `@jit.dont_look_inside`.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_fill_i(items: usize, length: usize, item: usize) {
    let mut i = 0;
    while i < length {
        ll_slice_setitem_fast_i(items, i, item);
        i += 1;
    }
}

/// `ll_reverse` on the `(items, length)` slice.  `ll_reverse`'s
/// `look_inside_iff(jit.isvirtual(l) ...)` asks about the GC list; the
/// items here are a raw address, which `jit.isvirtual` does not describe,
/// so the loop stays a residual call.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_reverse_i(items: usize, length: usize) {
    let length = length as isize;
    let mut i: isize = 0;
    let mut length_1_i = length - 1 - i;
    while i < length_1_i {
        let tmp = ll_slice_getitem_fast_i(items, i as usize);
        let other = ll_slice_getitem_fast_i(items, length_1_i as usize);
        ll_slice_setitem_fast_i(items, i as usize, other);
        ll_slice_setitem_fast_i(items, length_1_i as usize, tmp);
        i += 1;
        length_1_i -= 1;
    }
}

/// `<[T]>::rotate_right` on a pair slice. Three `ll_reverse` passes, the
/// same swap loop as [`ll_slice_reverse_i`]. The items are a raw address,
/// which `jit.isvirtual` does not describe, so the loop stays a residual
/// call — the same treatment as [`ll_slice_reverse_i`].
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_rotate_right_i(items: usize, length: usize, k: usize) {
    ll_slice_rotate_body!(items, length, k, ll_slice_reverse_range_i, true);
}

/// `<[T]>::rotate_left` on a pair slice.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_rotate_left_i(items: usize, length: usize, k: usize) {
    ll_slice_rotate_body!(items, length, k, ll_slice_reverse_range_i, false);
}

// ── `_r`: managed-reference items ───────────────────────────────────────

/// `ll_newemptylist`.
#[majit_macros::oopspec("newlist(0)")]
pub fn ll_vec_newemptylist_r() -> Vec<*mut u8> {
    Vec::new()
}

/// `lltypesystem/rlist.py ll_newlist`: `length` slots. Each slot is null
/// until the caller writes it, and `length` is already the list length.
/// No `newlist(length)` oopspec: the result is a raw `Vec` header (kind
/// int), and that rewrite emits GC `new_array_clear` into a ref bank.
pub fn ll_vec_newlist_r(length: usize) -> Vec<*mut u8> {
    let mut l = Vec::with_capacity(length);
    unsafe {
        l.set_len(length);
    }
    ll_vec_arrayclear_r(&mut l, length);
    l
}

/// `rlist.py _ll_zero_or_null` for a pointer item.
fn ll_vec_zero_or_null_r(item: *mut u8) -> bool {
    item.is_null()
}

/// `rgc.ll_arrayclear`. Writes null into each of the `count` slots.
#[majit_macros::dont_look_inside_cannot_raise]
fn ll_vec_arrayclear_r(l: &mut Vec<*mut u8>, count: usize) {
    let mut i = 0;
    while i < count {
        ll_vec_setitem_fast_r(l, i, std::ptr::null_mut());
        i += 1;
    }
}

/// `rlist.py _ll_alloc_and_clear`.
/// No `newlist_clear` oopspec: this is a raw `Vec`, not a GC list header.
pub fn ll_vec_alloc_and_clear_r(count: usize) -> Vec<*mut u8> {
    let mut l = ll_vec_newlist_r(count);
    ll_vec_arrayclear_r(&mut l, count);
    l
}

/// `@jit.look_inside_iff(lambda LIST, count, item: jit.isconstant(count) and count < 137)`.
fn ll_vec_alloc_and_set_nonnull_iff_r(count: usize, _item: *mut u8) -> bool {
    crate::jit::isconstant(&count) && count < 137
}

/// `rlist.py _ll_alloc_and_set_nonnull`.
#[majit_macros::look_inside_iff(ll_vec_alloc_and_set_nonnull_iff_r)]
pub fn ll_vec_alloc_and_set_nonnull_r(count: usize, item: *mut u8) -> Vec<*mut u8> {
    let mut l = ll_vec_newlist_r(count);
    let mut i = 0;
    while i < count {
        ll_vec_setitem_fast_r(&mut l, i, item);
        i += 1;
    }
    l
}

/// `rlist.py _ll_alloc_and_set_nojit`.
fn ll_vec_alloc_and_set_nojit_r(count: usize, item: *mut u8) -> Vec<*mut u8> {
    let mut l = ll_vec_newlist_r(count);
    if ll_vec_zero_or_null_r(item) {
        ll_vec_arrayclear_r(&mut l, count);
    } else {
        ll_slice_buffer_fill_r(ll_vec_items_r(&mut l), count, item);
    }
    l
}

/// `rlist.py _ll_alloc_and_set_jit`.
fn ll_vec_alloc_and_set_jit_r(count: usize, item: *mut u8) -> Vec<*mut u8> {
    if ll_vec_zero_or_null_r(item) {
        ll_vec_alloc_and_clear_r(count)
    } else {
        ll_vec_alloc_and_set_nonnull_r(count, item)
    }
}

/// `rlist.py ll_alloc_and_set`. `rarithmetic.int_force_ge_zero` is a no-op:
/// `count` is `usize`, already `>= 0`.
pub fn ll_vec_alloc_and_set_r(count: usize, item: *mut u8) -> Vec<*mut u8> {
    if crate::jit::we_are_jitted() {
        ll_vec_alloc_and_set_jit_r(count, item)
    } else {
        ll_vec_alloc_and_set_nojit_r(count, item)
    }
}

/// `ll_newlist_hint`.
#[majit_macros::oopspec("newlist_hint(lengthhint)")]
pub fn ll_vec_newlist_hint_r(lengthhint: usize) -> Vec<*mut u8> {
    Vec::with_capacity(lengthhint)
}

/// `ll_length`.
#[majit_macros::oopspec("list.len(l)")]
pub fn ll_vec_length_r(l: &mut Vec<*mut u8>) -> usize {
    vec_header_word(vec_header_r(l), VEC_LEN_WORD)
}

/// `ll_getitem_fast`: `index` is in bounds.
#[majit_macros::oopspec("list.getitem(l, index)")]
pub fn ll_vec_getitem_fast_r(l: &mut Vec<*mut u8>, index: usize) -> *mut u8 {
    raw_read_ptr(vec_item_addr(vec_header_r(l), index, ITEM_SIZE_R)) as *mut u8
}

/// `ll_setitem_fast`: `index` is in bounds.
#[majit_macros::oopspec("list.setitem(l, index, item)")]
pub fn ll_vec_setitem_fast_r(l: &mut Vec<*mut u8>, index: usize, item: *mut u8) {
    raw_write_ptr(
        vec_item_addr(vec_header_r(l), index, ITEM_SIZE_R),
        item as usize,
    )
}

/// `ll_append`.
pub fn ll_vec_append_r(l: &mut Vec<*mut u8>, newitem: *mut u8) {
    let length = ll_vec_length_r(l);
    ll_vec_resize_ge_r(l, length + 1);
    ll_vec_setitem_fast_r(l, length, newitem);
}

/// `_ll_list_resize_ge`.
pub fn ll_vec_resize_ge_r(l: &mut Vec<*mut u8>, newsize: usize) {
    let allocated = vec_header_word(vec_header_r(l), VEC_CAP_WORD);
    let cond = allocated < newsize;
    if crate::jit::isconstant(&allocated) && crate::jit::isconstant(&newsize) {
        if cond {
            ll_vec_resize_hint_really_r(l, newsize, true);
        }
    } else {
        crate::jit::conditional_call3(cond, ll_vec_resize_hint_really_r, &mut *l, newsize, true);
    }
    vec_set_header_word(vec_header_r(l), VEC_LEN_WORD, newsize);
}

/// `@jit.look_inside_iff(lambda l, newsize, overallocate:
/// jit.isconstant(len(l.items)) and jit.isconstant(newsize))`.
fn ll_vec_resize_hint_really_iff_r(
    l: &mut Vec<*mut u8>,
    newsize: usize,
    _overallocate: bool,
) -> bool {
    let allocated = vec_header_word(vec_header_r(l), VEC_CAP_WORD);
    crate::jit::isconstant(&allocated) && crate::jit::isconstant(&newsize)
}

/// `_ll_list_resize_hint_really`.
#[majit_macros::look_inside_iff(ll_vec_resize_hint_really_iff_r)]
pub fn ll_vec_resize_hint_really_r(l: &mut Vec<*mut u8>, newsize: usize, overallocate: bool) {
    let header = vec_header_r(l);
    let new_allocated = if newsize == 0 {
        vec_set_header_word(header, VEC_LEN_WORD, 0);
        0
    } else {
        vec_new_allocated(newsize, overallocate)
    };
    let items = vec_header_word(header, VEC_PTR_WORD);
    let allocated = vec_header_word(header, VEC_CAP_WORD);
    let newitems = vec_buf_realloc(items, allocated, new_allocated, ITEM_SIZE_R, ITEM_ALIGN_R);
    vec_set_header_word(header, VEC_PTR_WORD, newitems);
    vec_set_header_word(header, VEC_CAP_WORD, new_allocated);
}

/// `@jit.look_inside_iff(lambda l: jit.isvirtual(l) and
/// jit.isconstant(l.ll_length()))`.
fn ll_vec_reverse_iff_r(l: &mut Vec<*mut u8>) -> bool {
    let length = ll_vec_length_r(l);
    crate::jit::isvirtual(&*l) && crate::jit::isconstant(&length)
}

/// `ll_reverse`.
#[majit_macros::look_inside_iff(ll_vec_reverse_iff_r)]
pub fn ll_vec_reverse_r(l: &mut Vec<*mut u8>) {
    let length = ll_vec_length_r(l) as isize;
    let mut i: isize = 0;
    let mut length_1_i = length - 1 - i;
    while i < length_1_i {
        let tmp = ll_vec_getitem_fast_r(l, i as usize);
        let other = ll_vec_getitem_fast_r(l, length_1_i as usize);
        ll_vec_setitem_fast_r(l, i as usize, other);
        ll_vec_setitem_fast_r(l, length_1_i as usize, tmp);
        i += 1;
        length_1_i -= 1;
    }
}

/// `lltype.free(l, flavor='raw')` of a lowered `Vec` header and its buffer.
pub fn ll_vec_free_r(l: &mut Vec<*mut u8>) {
    let header = vec_header_r(l);
    let items = vec_header_word(header, VEC_PTR_WORD);
    let allocated = vec_header_word(header, VEC_CAP_WORD);
    vec_buf_free(items, allocated, ITEM_SIZE_R, ITEM_ALIGN_R);
    raw_free(header);
}

/// `ll_items`: the header's item pointer.
pub fn ll_vec_items_r(l: &mut Vec<*mut u8>) -> usize {
    vec_header_word(vec_header_r(l), VEC_PTR_WORD)
}

/// `ll_extend` from the `(items, length)` slice. `ovfcheck` on `len1 + len2`
/// rejects a wrapping sum, then `_ll_resize_ge` and `ll_arraycopy`.
pub fn ll_vec_extend_from_slice_r(l: &mut Vec<*mut u8>, items: usize, length: usize) {
    let len1 = ll_vec_length_r(l);
    let newlen = len1.checked_add(length).expect("Vec capacity overflow");
    ll_vec_resize_ge_r(l, newlen);
    ll_slice_arraycopy_r(items, ll_vec_items_r(l), 0, len1, length);
}

/// Length of a length-prefixed object GcArray. The array is a movable GC
/// object; callers re-read this from `l2` rather than caching it across a
/// collection.
fn gcarray_length_r(l2: *mut u8) -> usize {
    raw_read_ptr(raw_ptradd(l2 as usize, GCARRAY_LEN_OFFSET))
}

/// Item `index` of a length-prefixed object GcArray, read from the array
/// word so a collection that moves `l2` is visible on the next index.
fn gcarray_getitem_r(l2: *mut u8, index: usize) -> *mut u8 {
    raw_read_ptr(raw_ptradd(
        l2 as usize,
        GCARRAY_ITEMS_OFFSET + index * ITEM_SIZE_R,
    )) as *mut u8
}

/// `rlist.ll_extend(l1, l2)` when `l2` is a one-word object slice: the
/// GcArray word, not a `(items, length)` pair. Length is the array's
/// header; each item is read by index from `l2` after `_ll_resize_ge`,
/// which may collect.
///
/// `ovfcheck(len1 + len2)`: a wrapping sum would undersize the resize.
/// Overflow is MemoryError, the same `Vec capacity overflow` panic as
/// [`vec_buf_layout`].
pub fn ll_vec_extend_r(l: &mut Vec<*mut u8>, l2: *mut u8) {
    let len1 = ll_vec_length_r(l);
    let len2 = gcarray_length_r(l2);
    let newlength = len1
        .checked_add(len2)
        .unwrap_or_else(|| panic!("Vec capacity overflow"));
    ll_vec_resize_ge_r(l, newlength);
    let mut i = 0;
    while i < len2 {
        let item = gcarray_getitem_r(l2, i);
        ll_vec_setitem_fast_r(l, len1 + i, item);
        i += 1;
    }
}

/// `rgc.ll_arraycopy` over raw items. The items hold no GC pointer, so the
/// copy is the `raw_memcopy` of `length` items; `ll_arraycopy` is a residual
/// call (`list.ll_arraycopy`, `OS_ARRAYCOPY`), and so is this.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_arraycopy_r(
    source: usize,
    dest: usize,
    source_start: usize,
    dest_start: usize,
    length: usize,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            slice_item_addr(source, source_start, ITEM_SIZE_R) as *const u8,
            slice_item_addr(dest, dest_start, ITEM_SIZE_R) as *mut u8,
            item_bytes(length, ITEM_SIZE_R).expect("Vec capacity overflow"),
        );
    }
}

/// `ll_getitem_fast` on the items at `items`: `index` is in bounds.
pub fn ll_slice_getitem_fast_r(items: usize, index: usize) -> *mut u8 {
    raw_read_ptr(slice_item_addr(items, index, ITEM_SIZE_R)) as *mut u8
}

/// `ll_setitem_fast` on the items at `items`: `index` is in bounds.
pub fn ll_slice_setitem_fast_r(items: usize, index: usize, item: *mut u8) {
    raw_write_ptr(slice_item_addr(items, index, ITEM_SIZE_R), item as usize)
}

/// `ll_listcontains` on the `(items, length)` slice.  The items compare as
/// address words, so no reference read from the buffer meets `item`.
///
/// `ll_listcontains` is "not inlined by the JIT -- contains a loop".
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_contains_r(items: usize, length: usize, item: *mut u8) -> bool {
    let word = item as usize;
    let mut i = 0;
    while i < length {
        if raw_read_ptr(slice_item_addr(items, i, ITEM_SIZE_R)) == word {
            return true;
        }
        i += 1;
    }
    false
}

/// The item at `addr`, an [`ll_slice_get_addr_r`] result, or `default` when
/// `addr` is 0: `opt.copied().unwrap_or(default)`.
pub fn ll_slice_load_or_r(addr: usize, default: *mut u8) -> *mut u8 {
    let word = if addr != 0 {
        raw_read_ptr(addr)
    } else {
        default as usize
    };
    word as *mut u8
}

/// `s.to_vec()`: `ll_copy` — a new `Vec` of `length` items, then
/// `ll_arraycopy` of the items at `items`.
pub fn ll_slice_to_vec_r(items: usize, length: usize) -> Vec<*mut u8> {
    let mut l = ll_vec_newlist_hint_r(length);
    ll_vec_resize_ge_r(&mut l, length);
    ll_slice_arraycopy_r(items, ll_vec_items_r(&mut l), 0, 0, length);
    l
}

/// The item pointer of `&s[start..]` over the items at `items`: `start` is
/// at most the slice's length.
pub fn ll_slice_offset_r(items: usize, start: usize) -> usize {
    slice_item_addr(items, start, ITEM_SIZE_R)
}

/// The address of item `index` of `(items, length)`, or 0 when `index` is
/// out of bounds: the one-word `Option<&T>` of `s.get(index)`.
pub fn ll_slice_get_addr_r(items: usize, length: usize, index: usize) -> usize {
    if index < length {
        slice_item_addr(items, index, ITEM_SIZE_R)
    } else {
        0
    }
}

/// `lltype.malloc(Array(ITEM), length, flavor='raw')`: the item buffer of an
/// array the lowering keeps in raw memory because it is borrowed as a slice.
/// Opaque like [`vec_buf_alloc`]. `checked_mul` overflow stays in this
/// residual; `raw_malloc_varsize_char` is already dont_look_inside.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_new_r(length: usize) -> usize {
    let size = item_bytes(length, ITEM_SIZE_R).expect("Vec capacity overflow");
    raw_malloc_varsize_char(size)
}

/// `[item; length]` into the buffer at `items`.
/// `rgc.ll_arrayfill`, which is `@jit.dont_look_inside`.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_fill_r(items: usize, length: usize, item: *mut u8) {
    let mut i = 0;
    while i < length {
        ll_slice_setitem_fast_r(items, i, item);
        i += 1;
    }
}

/// `ll_reverse` on the `(items, length)` slice.  `ll_reverse`'s
/// `look_inside_iff(jit.isvirtual(l) ...)` asks about the GC list; the
/// items here are a raw address, which `jit.isvirtual` does not describe,
/// so the loop stays a residual call.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_reverse_r(items: usize, length: usize) {
    let length = length as isize;
    let mut i: isize = 0;
    let mut length_1_i = length - 1 - i;
    while i < length_1_i {
        // The items move as address words.
        let low = slice_item_addr(items, i as usize, ITEM_SIZE_R);
        let high = slice_item_addr(items, length_1_i as usize, ITEM_SIZE_R);
        let tmp = raw_read_ptr(low);
        raw_write_ptr(low, raw_read_ptr(high));
        raw_write_ptr(high, tmp);
        i += 1;
        length_1_i -= 1;
    }
}

/// `<[T]>::rotate_right` on a pair slice. Three `ll_reverse` passes, the
/// same address-word swap as [`ll_slice_reverse_r`].
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_rotate_right_r(items: usize, length: usize, k: usize) {
    ll_slice_rotate_body!(items, length, k, ll_slice_reverse_range_r, true);
}

/// `<[T]>::rotate_left` on a pair slice.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_rotate_left_r(items: usize, length: usize, k: usize) {
    ll_slice_rotate_body!(items, length, k, ll_slice_reverse_range_r, false);
}

// ── `_f`: `f64` items ───────────────────────────────────────────────────

/// `ll_newemptylist`.
#[majit_macros::oopspec("newlist(0)")]
pub fn ll_vec_newemptylist_f() -> Vec<f64> {
    Vec::new()
}

/// `lltypesystem/rlist.py ll_newlist`: `length` slots. Each slot is `0.0`
/// until the caller writes it, and `length` is already the list length.
/// No `newlist(length)` oopspec: the result is a raw `Vec` header (kind
/// int), and that rewrite emits GC `new_array_clear` into a ref bank.
pub fn ll_vec_newlist_f(length: usize) -> Vec<f64> {
    let mut l = Vec::with_capacity(length);
    unsafe {
        l.set_len(length);
    }
    ll_vec_arrayclear_f(&mut l, length);
    l
}

/// `rlist.py _ll_zero_or_null`: `not` of the widened number.
/// Both `0.0` and `-0.0` are zero.
fn ll_vec_zero_or_null_f(item: f64) -> bool {
    item == 0.0
}

/// `rgc.ll_arrayclear`. Writes `0.0` into each of the `count` slots.
#[majit_macros::dont_look_inside_cannot_raise]
fn ll_vec_arrayclear_f(l: &mut Vec<f64>, count: usize) {
    let mut i = 0;
    while i < count {
        ll_vec_setitem_fast_f(l, i, 0.0);
        i += 1;
    }
}

/// `rlist.py _ll_alloc_and_clear`.
/// No `newlist_clear` oopspec: this is a raw `Vec`, not a GC list header.
pub fn ll_vec_alloc_and_clear_f(count: usize) -> Vec<f64> {
    let mut l = ll_vec_newlist_f(count);
    ll_vec_arrayclear_f(&mut l, count);
    l
}

/// `@jit.look_inside_iff(lambda LIST, count, item: jit.isconstant(count) and count < 137)`.
fn ll_vec_alloc_and_set_nonnull_iff_f(count: usize, _item: f64) -> bool {
    crate::jit::isconstant(&count) && count < 137
}

/// `rlist.py _ll_alloc_and_set_nonnull`.
#[majit_macros::look_inside_iff(ll_vec_alloc_and_set_nonnull_iff_f)]
pub fn ll_vec_alloc_and_set_nonnull_f(count: usize, item: f64) -> Vec<f64> {
    let mut l = ll_vec_newlist_f(count);
    let mut i = 0;
    while i < count {
        ll_vec_setitem_fast_f(&mut l, i, item);
        i += 1;
    }
    l
}

/// `rlist.py _ll_alloc_and_set_nojit`.
fn ll_vec_alloc_and_set_nojit_f(count: usize, item: f64) -> Vec<f64> {
    let mut l = ll_vec_newlist_f(count);
    if ll_vec_zero_or_null_f(item) {
        ll_vec_arrayclear_f(&mut l, count);
    } else {
        ll_slice_buffer_fill_f(ll_vec_items_f(&mut l), count, item);
    }
    l
}

/// `rlist.py _ll_alloc_and_set_jit`.
fn ll_vec_alloc_and_set_jit_f(count: usize, item: f64) -> Vec<f64> {
    if ll_vec_zero_or_null_f(item) {
        ll_vec_alloc_and_clear_f(count)
    } else {
        ll_vec_alloc_and_set_nonnull_f(count, item)
    }
}

/// `rlist.py ll_alloc_and_set`. `rarithmetic.int_force_ge_zero` is a no-op:
/// `count` is `usize`, already `>= 0`.
pub fn ll_vec_alloc_and_set_f(count: usize, item: f64) -> Vec<f64> {
    if crate::jit::we_are_jitted() {
        ll_vec_alloc_and_set_jit_f(count, item)
    } else {
        ll_vec_alloc_and_set_nojit_f(count, item)
    }
}

/// `ll_newlist_hint`.
#[majit_macros::oopspec("newlist_hint(lengthhint)")]
pub fn ll_vec_newlist_hint_f(lengthhint: usize) -> Vec<f64> {
    Vec::with_capacity(lengthhint)
}

/// `ll_length`.
#[majit_macros::oopspec("list.len(l)")]
pub fn ll_vec_length_f(l: &mut Vec<f64>) -> usize {
    vec_header_word(vec_header_f(l), VEC_LEN_WORD)
}

/// `ll_getitem_fast`: `index` is in bounds.
#[majit_macros::oopspec("list.getitem(l, index)")]
pub fn ll_vec_getitem_fast_f(l: &mut Vec<f64>, index: usize) -> f64 {
    raw_read_f64(vec_item_addr(vec_header_f(l), index, ITEM_SIZE_F))
}

/// `ll_setitem_fast`: `index` is in bounds.
#[majit_macros::oopspec("list.setitem(l, index, item)")]
pub fn ll_vec_setitem_fast_f(l: &mut Vec<f64>, index: usize, item: f64) {
    raw_write_f64(vec_item_addr(vec_header_f(l), index, ITEM_SIZE_F), item)
}

/// `ll_append`.
pub fn ll_vec_append_f(l: &mut Vec<f64>, newitem: f64) {
    let length = ll_vec_length_f(l);
    ll_vec_resize_ge_f(l, length + 1);
    ll_vec_setitem_fast_f(l, length, newitem);
}

/// `_ll_list_resize_ge`.
pub fn ll_vec_resize_ge_f(l: &mut Vec<f64>, newsize: usize) {
    let allocated = vec_header_word(vec_header_f(l), VEC_CAP_WORD);
    let cond = allocated < newsize;
    if crate::jit::isconstant(&allocated) && crate::jit::isconstant(&newsize) {
        if cond {
            ll_vec_resize_hint_really_f(l, newsize, true);
        }
    } else {
        crate::jit::conditional_call3(cond, ll_vec_resize_hint_really_f, &mut *l, newsize, true);
    }
    vec_set_header_word(vec_header_f(l), VEC_LEN_WORD, newsize);
}

/// `@jit.look_inside_iff(lambda l, newsize, overallocate:
/// jit.isconstant(len(l.items)) and jit.isconstant(newsize))`.
fn ll_vec_resize_hint_really_iff_f(l: &mut Vec<f64>, newsize: usize, _overallocate: bool) -> bool {
    let allocated = vec_header_word(vec_header_f(l), VEC_CAP_WORD);
    crate::jit::isconstant(&allocated) && crate::jit::isconstant(&newsize)
}

/// `_ll_list_resize_hint_really`.
#[majit_macros::look_inside_iff(ll_vec_resize_hint_really_iff_f)]
pub fn ll_vec_resize_hint_really_f(l: &mut Vec<f64>, newsize: usize, overallocate: bool) {
    let header = vec_header_f(l);
    let new_allocated = if newsize == 0 {
        vec_set_header_word(header, VEC_LEN_WORD, 0);
        0
    } else {
        vec_new_allocated(newsize, overallocate)
    };
    let items = vec_header_word(header, VEC_PTR_WORD);
    let allocated = vec_header_word(header, VEC_CAP_WORD);
    let newitems = vec_buf_realloc(items, allocated, new_allocated, ITEM_SIZE_F, ITEM_ALIGN_F);
    vec_set_header_word(header, VEC_PTR_WORD, newitems);
    vec_set_header_word(header, VEC_CAP_WORD, new_allocated);
}

/// `@jit.look_inside_iff(lambda l: jit.isvirtual(l) and
/// jit.isconstant(l.ll_length()))`.
fn ll_vec_reverse_iff_f(l: &mut Vec<f64>) -> bool {
    let length = ll_vec_length_f(l);
    crate::jit::isvirtual(&*l) && crate::jit::isconstant(&length)
}

/// `ll_reverse`.
#[majit_macros::look_inside_iff(ll_vec_reverse_iff_f)]
pub fn ll_vec_reverse_f(l: &mut Vec<f64>) {
    let length = ll_vec_length_f(l) as isize;
    let mut i: isize = 0;
    let mut length_1_i = length - 1 - i;
    while i < length_1_i {
        let tmp = ll_vec_getitem_fast_f(l, i as usize);
        let other = ll_vec_getitem_fast_f(l, length_1_i as usize);
        ll_vec_setitem_fast_f(l, i as usize, other);
        ll_vec_setitem_fast_f(l, length_1_i as usize, tmp);
        i += 1;
        length_1_i -= 1;
    }
}

/// `lltype.free(l, flavor='raw')` of a lowered `Vec` header and its buffer.
pub fn ll_vec_free_f(l: &mut Vec<f64>) {
    let header = vec_header_f(l);
    let items = vec_header_word(header, VEC_PTR_WORD);
    let allocated = vec_header_word(header, VEC_CAP_WORD);
    vec_buf_free(items, allocated, ITEM_SIZE_F, ITEM_ALIGN_F);
    raw_free(header);
}

/// `ll_items`: the header's item pointer.
pub fn ll_vec_items_f(l: &mut Vec<f64>) -> usize {
    vec_header_word(vec_header_f(l), VEC_PTR_WORD)
}

/// `ll_extend` from the `(items, length)` slice. `ovfcheck` on `len1 + len2`
/// rejects a wrapping sum, then `_ll_resize_ge` and `ll_arraycopy`.
pub fn ll_vec_extend_from_slice_f(l: &mut Vec<f64>, items: usize, length: usize) {
    let len1 = ll_vec_length_f(l);
    let newlen = len1.checked_add(length).expect("Vec capacity overflow");
    ll_vec_resize_ge_f(l, newlen);
    ll_slice_arraycopy_f(items, ll_vec_items_f(l), 0, len1, length);
}

/// `rgc.ll_arraycopy` over raw items. The items hold no GC pointer, so the
/// copy is the `raw_memcopy` of `length` items; `ll_arraycopy` is a residual
/// call (`list.ll_arraycopy`, `OS_ARRAYCOPY`), and so is this.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_arraycopy_f(
    source: usize,
    dest: usize,
    source_start: usize,
    dest_start: usize,
    length: usize,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            slice_item_addr(source, source_start, ITEM_SIZE_F) as *const u8,
            slice_item_addr(dest, dest_start, ITEM_SIZE_F) as *mut u8,
            item_bytes(length, ITEM_SIZE_F).expect("Vec capacity overflow"),
        );
    }
}

/// `ll_getitem_fast` on the items at `items`: `index` is in bounds.
pub fn ll_slice_getitem_fast_f(items: usize, index: usize) -> f64 {
    raw_read_f64(slice_item_addr(items, index, ITEM_SIZE_F))
}

/// `ll_setitem_fast` on the items at `items`: `index` is in bounds.
pub fn ll_slice_setitem_fast_f(items: usize, index: usize, item: f64) {
    raw_write_f64(slice_item_addr(items, index, ITEM_SIZE_F), item)
}

/// `ll_listcontains` on the `(items, length)` slice.
///
/// `ll_listcontains` is "not inlined by the JIT -- contains a loop".
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_contains_f(items: usize, length: usize, item: f64) -> bool {
    let mut i = 0;
    while i < length {
        if ll_slice_getitem_fast_f(items, i) == item {
            return true;
        }
        i += 1;
    }
    false
}

/// The item at `addr`, an [`ll_slice_get_addr_f`] result, or `default` when
/// `addr` is 0: `opt.copied().unwrap_or(default)`.
pub fn ll_slice_load_or_f(addr: usize, default: f64) -> f64 {
    if addr != 0 {
        ll_slice_getitem_fast_f(addr, 0)
    } else {
        default
    }
}

/// `s.to_vec()`: `ll_copy` — a new `Vec` of `length` items, then
/// `ll_arraycopy` of the items at `items`.
pub fn ll_slice_to_vec_f(items: usize, length: usize) -> Vec<f64> {
    let mut l = ll_vec_newlist_hint_f(length);
    ll_vec_resize_ge_f(&mut l, length);
    ll_slice_arraycopy_f(items, ll_vec_items_f(&mut l), 0, 0, length);
    l
}

/// The item pointer of `&s[start..]` over the items at `items`: `start` is
/// at most the slice's length.
pub fn ll_slice_offset_f(items: usize, start: usize) -> usize {
    slice_item_addr(items, start, ITEM_SIZE_F)
}

/// The address of item `index` of `(items, length)`, or 0 when `index` is
/// out of bounds: the one-word `Option<&T>` of `s.get(index)`.
pub fn ll_slice_get_addr_f(items: usize, length: usize, index: usize) -> usize {
    if index < length {
        slice_item_addr(items, index, ITEM_SIZE_F)
    } else {
        0
    }
}

/// `lltype.malloc(Array(ITEM), length, flavor='raw')`: the item buffer of an
/// array the lowering keeps in raw memory because it is borrowed as a slice.
/// Opaque like [`vec_buf_alloc`]. `checked_mul` overflow stays in this
/// residual; `raw_malloc_varsize_char` is already dont_look_inside.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_new_f(length: usize) -> usize {
    let size = item_bytes(length, ITEM_SIZE_F).expect("Vec capacity overflow");
    raw_malloc_varsize_char(size)
}

/// `[item; length]` into the buffer at `items`.
/// `rgc.ll_arrayfill`, which is `@jit.dont_look_inside`.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_buffer_fill_f(items: usize, length: usize, item: f64) {
    let mut i = 0;
    while i < length {
        ll_slice_setitem_fast_f(items, i, item);
        i += 1;
    }
}

/// `ll_reverse` on the `(items, length)` slice.  `ll_reverse`'s
/// `look_inside_iff(jit.isvirtual(l) ...)` asks about the GC list; the
/// items here are a raw address, which `jit.isvirtual` does not describe,
/// so the loop stays a residual call.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_reverse_f(items: usize, length: usize) {
    let length = length as isize;
    let mut i: isize = 0;
    let mut length_1_i = length - 1 - i;
    while i < length_1_i {
        let tmp = ll_slice_getitem_fast_f(items, i as usize);
        let other = ll_slice_getitem_fast_f(items, length_1_i as usize);
        ll_slice_setitem_fast_f(items, i as usize, other);
        ll_slice_setitem_fast_f(items, length_1_i as usize, tmp);
        i += 1;
        length_1_i -= 1;
    }
}

/// `<[T]>::rotate_right` on a pair slice. Three `ll_reverse` passes, the
/// same swap loop as [`ll_slice_reverse_f`].
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_rotate_right_f(items: usize, length: usize, k: usize) {
    ll_slice_rotate_body!(items, length, k, ll_slice_reverse_range_f, true);
}

/// `<[T]>::rotate_left` on a pair slice.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn ll_slice_rotate_left_f(items: usize, length: usize, k: usize) {
    ll_slice_rotate_body!(items, length, k, ll_slice_reverse_range_f, false);
}

#[cfg(test)]
mod tests {
    use super::super::rffi::raw_malloc_varsize_char;
    use super::*;

    /// The oopspec helpers are callable residuals: a by-value `Vec` result
    /// comes back as the address of a header `ll_vec_free_*` releases, and a
    /// `&mut Vec` argument takes that address.
    #[test]
    fn oopspec_helpers_have_word_abi_call_targets() {
        let header = __majit_call_target_ll_vec_newlist_hint_r(4);
        assert_ne!(header, 0);
        assert_eq!(__majit_call_target_ll_vec_length_r(header), 0);
        let l = unsafe { &mut *(header as usize as *mut Vec<*mut u8>) };
        assert!(l.capacity() >= 4);
        l.push(8 as *mut u8);
        assert_eq!(__majit_call_target_ll_vec_length_r(header), 1);
        ll_vec_free_r(l);
    }

    #[test]
    #[should_panic(expected = "Vec capacity overflow")]
    fn slice_buffer_new_rejects_a_wrapping_length() {
        let _ = ll_slice_buffer_new_i(usize::MAX);
    }

    #[test]
    #[should_panic(expected = "Vec capacity overflow")]
    fn extend_from_slice_rejects_a_wrapping_length() {
        let mut l = ll_vec_newlist_i(1);
        ll_vec_extend_from_slice_i(&mut l, 0, usize::MAX);
    }

    #[test]
    fn allocator_paths_name_these_functions() {
        let module = module_path!().trim_end_matches("::tests");
        assert_eq!(
            majit_ir::rvec::VEC_BUF_ALLOC,
            format!("{module}::vec_buf_alloc")
        );
        assert_eq!(
            majit_ir::rvec::VEC_BUF_ALLOC_CLEAR,
            format!("{module}::vec_buf_alloc_clear")
        );
        let rffi = module.replace("::rvec", "::rffi");
        assert_eq!(
            majit_ir::rvec::VEC_HEADER_MALLOC,
            format!("{rffi}::raw_malloc_varsize_char")
        );
    }
    use majit_ir::rvec::{
        SliceOp, VEC_HEADER_WORDS, VecItemKind, VecOp, slice_helper_path, vec_helper_path,
    };

    #[test]
    fn alloc_and_set_fills_zero_and_nonzero_items() {
        assert_eq!(ll_vec_alloc_and_set_i(0, 0), Vec::<usize>::new());
        assert_eq!(ll_vec_alloc_and_set_i(0, 7), Vec::<usize>::new());
        assert_eq!(ll_vec_alloc_and_set_i(3, 0), vec![0usize; 3]);
        assert_eq!(ll_vec_alloc_and_set_i(3, 7), vec![7usize; 3]);
        assert_eq!(ll_vec_alloc_and_clear_i(0), Vec::<usize>::new());
        assert_eq!(ll_vec_alloc_and_clear_i(3), vec![0usize; 3]);

        let p = 0x20 as *mut u8;
        assert_eq!(ll_vec_alloc_and_set_r(0, std::ptr::null_mut()), Vec::new());
        assert_eq!(ll_vec_alloc_and_set_r(0, p), Vec::new());
        assert_eq!(
            ll_vec_alloc_and_set_r(3, std::ptr::null_mut()),
            vec![std::ptr::null_mut(); 3]
        );
        assert_eq!(ll_vec_alloc_and_set_r(3, p), vec![p; 3]);
        assert_eq!(ll_vec_alloc_and_clear_r(0), Vec::<*mut u8>::new());
        assert_eq!(ll_vec_alloc_and_clear_r(3), vec![std::ptr::null_mut(); 3]);

        assert_eq!(ll_vec_alloc_and_set_f(0, 0.0), Vec::<f64>::new());
        assert_eq!(ll_vec_alloc_and_set_f(0, 1.5), Vec::<f64>::new());
        assert_eq!(ll_vec_alloc_and_set_f(3, 0.0), vec![0.0; 3]);
        assert!(
            ll_vec_alloc_and_set_f(2, -0.0)
                .iter()
                .all(|item| item.to_bits() == 0)
        );
        assert_eq!(ll_vec_alloc_and_set_f(3, 1.5), vec![1.5; 3]);
        assert_eq!(ll_vec_alloc_and_clear_f(0), Vec::<f64>::new());
        assert_eq!(ll_vec_alloc_and_clear_f(3), vec![0.0; 3]);
    }

    #[test]
    fn as_ptr_reads_the_header_buffer_pointer_word() {
        let mut bytes: Vec<u8> = vec![1, 2, 3];
        let header = &mut bytes as *mut Vec<u8> as usize;
        assert_eq!(ll_vec_as_ptr(header), bytes.as_ptr() as usize);
        let mut words: Vec<usize> = vec![7, 8];
        let header = &mut words as *mut Vec<usize> as usize;
        assert_eq!(ll_vec_as_ptr(header), words.as_ptr() as usize);
        assert_eq!(ll_vec_as_ptr(header), ll_vec_items_i(&mut words));
    }

    #[test]
    fn helpers_grow_and_reverse_a_host_vec() {
        let mut v = ll_vec_newemptylist_i();
        for item in 0..40 {
            ll_vec_append_i(&mut v, item);
        }
        assert_eq!(ll_vec_length_i(&mut v), 40);
        assert_eq!(v, (0..40).collect::<Vec<_>>());
        assert!(v.capacity() >= 40);
        ll_vec_reverse_i(&mut v);
        assert_eq!(v, (0..40).rev().collect::<Vec<_>>());
        ll_vec_setitem_fast_i(&mut v, 3, 99);
        assert_eq!(ll_vec_getitem_fast_i(&mut v, 3), 99);
        // The host drop frees the buffer the helpers grew.
        drop(v);

        let mut f = ll_vec_newlist_hint_f(2);
        for item in 0..5 {
            ll_vec_append_f(&mut f, item as f64 + 0.5);
        }
        ll_vec_reverse_f(&mut f);
        assert_eq!(f, vec![4.5, 3.5, 2.5, 1.5, 0.5]);

        let mut r = ll_vec_newemptylist_r();
        let a = 0x10 as *mut u8;
        let b = 0x20 as *mut u8;
        ll_vec_append_r(&mut r, a);
        ll_vec_append_r(&mut r, b);
        ll_vec_reverse_r(&mut r);
        assert_eq!(r, vec![b, a]);
    }

    #[test]
    fn helpers_rotate_a_pair_slice() {
        let mut v: Vec<usize> = (0..5).collect();
        let items = ll_vec_items_i(&mut v);
        let n = ll_vec_length_i(&mut v);
        ll_slice_rotate_right_i(items, n, 2);
        assert_eq!(v, vec![3, 4, 0, 1, 2]);
        ll_slice_rotate_left_i(items, n, 2);
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
        ll_slice_rotate_right_i(items, n, 0);
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
        ll_slice_rotate_right_i(items, n, 5);
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
        ll_slice_rotate_left_i(items, n, 7);
        assert_eq!(v, vec![2, 3, 4, 0, 1]);

        ll_slice_rotate_right_i(0, 0, 3);

        let a = 0x10 as *mut u8;
        let b = 0x20 as *mut u8;
        let c = 0x30 as *mut u8;
        let mut r = vec![a, b, c];
        let items = ll_vec_items_r(&mut r);
        ll_slice_rotate_right_r(items, 3, 1);
        assert_eq!(r, vec![c, a, b]);
        ll_slice_rotate_left_r(items, 3, 1);
        assert_eq!(r, vec![a, b, c]);

        let mut f = vec![1.5, 2.5, 3.5, 4.5];
        let items = ll_vec_items_f(&mut f);
        ll_slice_rotate_left_f(items, 4, 1);
        assert_eq!(f, vec![2.5, 3.5, 4.5, 1.5]);
        ll_slice_rotate_right_f(items, 4, 1);
        assert_eq!(f, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn free_releases_a_raw_header_and_its_buffer() {
        let header = raw_malloc_varsize_char(VEC_HEADER_WORDS * WORD);
        unsafe { (header as *mut Vec<usize>).write(Vec::new()) };
        let l = unsafe { &mut *(header as *mut Vec<usize>) };
        for item in 0..10 {
            ll_vec_append_i(l, item);
        }
        assert_eq!(ll_vec_length_i(l), 10);
        ll_vec_free_i(l);
    }

    #[test]
    fn slice_helpers_read_and_write_through_the_items_pointer() {
        let mut v: Vec<usize> = (0..6).collect();
        let items = ll_vec_items_i(&mut v);
        assert_eq!(items, v.as_ptr() as usize);
        assert_eq!(ll_slice_getitem_fast_i(items, 4), 4);
        ll_slice_setitem_fast_i(items, 4, 40);
        ll_slice_reverse_i(items, ll_vec_length_i(&mut v));
        assert_eq!(v, vec![5, 40, 3, 2, 1, 0]);

        let floats = [1.5f64, 2.5, 3.5];
        assert_eq!(ll_slice_getitem_fast_f(floats.as_ptr() as usize, 2), 3.5);

        let mut r = vec![0x10 as *mut u8, 0x20 as *mut u8];
        let items = ll_vec_items_r(&mut r);
        ll_slice_reverse_r(items, 2);
        assert_eq!(ll_slice_getitem_fast_r(items, 0), 0x20 as *mut u8);

        ll_slice_reverse_i(0, 0);

        let slot = ll_slice_len_slot_new();
        ll_slice_len_slot_store(slot, 17);
        assert_eq!(ll_slice_len_slot_take(slot), 17);
    }

    #[test]
    fn slice_buffer_holds_an_array_borrowed_as_a_slice() {
        let items = ll_slice_buffer_new_i(3);
        ll_slice_buffer_fill_i(items, 3, 7);
        ll_slice_setitem_fast_i(items, 1, 9);
        assert!(ll_slice_contains_i(items, 3, 9));
        assert!(!ll_slice_contains_i(items, 3, 8));
        assert_eq!(ll_slice_getitem_fast_i(items, 2), 7);
        let second = ll_slice_get_addr_i(items, 3, 1);
        assert_ne!(second, 0);
        assert_eq!(ll_slice_getitem_fast_i(second, 0), 9);
        assert_eq!(ll_slice_get_addr_i(items, 3, 3), 0);
        assert_eq!(ll_slice_get_addr_i(items, 0, usize::MAX), 0);
        assert_eq!(ll_slice_getitem_fast_i(ll_slice_offset_i(items, 1), 0), 9);
        assert_eq!(ll_slice_load_or_i(second, 5), 9);
        assert_eq!(ll_slice_load_or_i(0, 5), 5);
        assert_eq!(ll_slice_to_vec_i(items, 3), vec![7, 9, 7]);
        ll_slice_buffer_free(items);

        let floats = ll_slice_buffer_new_f(2);
        ll_slice_buffer_fill_f(floats, 2, 0.5);
        assert!(ll_slice_contains_f(floats, 2, 0.5));
        ll_slice_buffer_free(floats);

        let empty = ll_slice_buffer_new_r(0);
        assert!(!ll_slice_contains_r(empty, 0, std::ptr::null_mut()));
        ll_slice_buffer_free(empty);
    }

    #[test]
    fn resize_to_zero_empties_the_vec() {
        let mut v: Vec<usize> = (0..5).collect();
        ll_vec_resize_hint_really_i(&mut v, 0, false);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn residual_entries_are_published() {
        let module = module_path!().trim_end_matches("::tests");
        let mut published = Vec::new();
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|desc| published.push(desc.path));
        for leaf in [
            "vec_buf_alloc",
            "vec_buf_alloc_clear",
            "vec_buf_realloc",
            "vec_buf_free",
            "ll_vec_resize_hint_really_i",
            "ll_vec_resize_hint_really_r",
            "ll_vec_resize_hint_really_f",
            "ll_vec_reverse_i",
            "ll_slice_reverse_i",
            "ll_slice_reverse_r",
            "ll_slice_reverse_f",
            "ll_slice_rotate_left_i",
            "ll_slice_rotate_right_r",
            "ll_slice_rotate_left_f",
            "ll_slice_bounds_panic",
        ] {
            let path = format!("{module}::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }

    #[test]
    fn extend_copies_gcarray_items_after_resize() {
        #[repr(C)]
        struct GcArrayWord {
            length: usize,
            items: [*mut u8; 2],
        }
        let mut src = GcArrayWord {
            length: 2,
            items: [0x10 as *mut u8, 0x20 as *mut u8],
        };
        let mut l = ll_vec_newemptylist_r();
        ll_vec_append_r(&mut l, 0x01 as *mut u8);
        ll_vec_extend_r(&mut l, &mut src as *mut GcArrayWord as *mut u8);
        assert_eq!(ll_vec_length_r(&mut l), 3);
        assert_eq!(ll_vec_getitem_fast_r(&mut l, 0), 0x01 as *mut u8);
        assert_eq!(ll_vec_getitem_fast_r(&mut l, 1), 0x10 as *mut u8);
        assert_eq!(ll_vec_getitem_fast_r(&mut l, 2), 0x20 as *mut u8);

        let mut empty = GcArrayWord {
            length: 0,
            items: [std::ptr::null_mut(), std::ptr::null_mut()],
        };
        let before = ll_vec_length_r(&mut l);
        ll_vec_extend_r(&mut l, &mut empty as *mut GcArrayWord as *mut u8);
        assert_eq!(ll_vec_length_r(&mut l), before);
    }

    /// `ovfcheck(len1 + len2)` in `ll_extend`: a wrapping sum is MemoryError,
    /// the same panic [`vec_buf_layout`] uses for a capacity overflow.
    #[test]
    #[should_panic(expected = "Vec capacity overflow")]
    fn extend_overflowing_length_sum_is_memory_error() {
        #[repr(C)]
        struct GcArrayWord {
            length: usize,
            items: [*mut u8; 1],
        }
        let mut src = GcArrayWord {
            length: usize::MAX,
            items: [std::ptr::null_mut()],
        };
        let mut l = ll_vec_newemptylist_r();
        ll_vec_append_r(&mut l, 0x01 as *mut u8);
        ll_vec_extend_r(&mut l, &mut src as *mut GcArrayWord as *mut u8);
    }

    #[test]
    fn table_paths_name_this_module() {
        let module = module_path!().trim_end_matches("::tests");
        for kind in VecItemKind::ALL {
            let paths = VecOp::ALL
                .into_iter()
                .map(|op| vec_helper_path(op, kind))
                .chain(
                    SliceOp::ALL
                        .into_iter()
                        .map(|op| slice_helper_path(op, kind)),
                );
            for path in paths {
                assert_eq!(
                    path.rsplit_once("::").map(|(m, _)| m),
                    Some(module),
                    "{path}"
                );
            }
        }
    }
}
