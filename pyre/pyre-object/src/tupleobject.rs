//! W_TupleObject — Python `tuple` type.
//!
//! Upstream `pypy/objspace/std/tupleobject.py` `W_TupleObject`
//! stores `wrappeditems: list` with `_immutable_fields_ =
//! ['wrappeditems[*]']`. After RPython translation `wrappeditems`
//! becomes `Ptr(GcArray(OBJECTPTR))`; the `[*]` annotation marks both
//! the list and its contents as immutable so the JIT can hoist loads.
//! pyre stores the array via `*mut ItemsBlock` (shared GcArray body
//! layer with `W_ListObject`) and the content hash in `_hash_cache`.
//! Length comes directly from the
//! GcArray header — `arraylen_gc(wrappeditems)` on the JIT side and
//! `items_block_capacity(wrappeditems)` on the host side.
//! Python 3.14 adds `PyTupleObject.ob_hash`, initially `-1`, and caches the
//! aggregate after the first successful element walk.  `hash` is the
//! corresponding field in pyre; it is the deliberate 3.14 semantic delta
//! from PyPy's otherwise authoritative `W_TupleObject` layout.
//!
//! Arity-2 tuples are routed through specialised variants
//! (`W_SpecialisedTupleObject_{ii,ff,oo}` per
//! `pypy/objspace/std/specialisedtupleobject.py`) — see
//! `makespecialisedtuple2` below. Polymorphic readers
//! (`w_tuple_len`, `w_tuple_getitem`, `w_tuple_items_copy_as_vec`)
//! dispatch on `ob_type` so callers see a uniform tuple API.
//!
//! Tuples are immutable after creation. `wrappeditems` is allocated
//! once via `alloc_tuple_items_block` (exact-size; empty tuple yields
//! a 0-cap header-only block) and never resized.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::floatobject::{w_float_get_value, w_float_new};
use crate::intobject::w_int_new;
use crate::listobject::{is_plain_int1, plain_int_w};
use crate::object_array::{
    ITEMS_BLOCK_ITEMS_OFFSET, ItemsBlock, alloc_tuple_items_block_gc, items_block_capacity,
};

/// Tuple `wrappeditems` is the fixed object array. Its callers must not share
/// `items_block_items_base`, whose other arguments are resizable list headers.
unsafe fn tuple_items_base(block: *mut ItemsBlock) -> *mut PyObjectRef {
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET) as *mut PyObjectRef }
}
use crate::pyobject::*;
use crate::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    W_SpecialisedTupleObject_ff, W_SpecialisedTupleObject_ii, W_SpecialisedTupleObject_oo,
    is_specialised_tuple_ff, is_specialised_tuple_ii, is_specialised_tuple_oo,
    w_specialised_tuple_ff_new, w_specialised_tuple_ii_new, w_specialised_tuple_oo_new,
};
use std::sync::atomic::{AtomicI64, Ordering};

/// `ob_hash` before the first successful element walk, and the one value a
/// finished walk never stores — `tuplehash` remaps a computed `-1` to
/// `1546275796`, so a cache hit is exactly `hash != TUPLE_HASH_UNSET`.
///
/// `_PyTuple_RESET_HASH_CACHE` is the single place this is written upstream.
/// Here every constructor writes it instead, host and trace alike: zeroed
/// allocation memory would otherwise read back as the valid hash `0`.
pub const TUPLE_HASH_UNSET: i64 = -1;

/// Python tuple object — array-backed default representation.
///
/// Layout mirrors `pypy/objspace/std/tupleobject.py W_TupleObject` after
/// RPython translation:
/// `{wrappeditems: Ptr(GcArray(OBJECTPTR)), _hash_cache: Signed}`.
/// `_immutable_fields_ = ['wrappeditems[*]']`: the field read is pure,
/// and so is every item read through the array it produced; the array
/// length comes from `arraylen_gc` against the GcArray header.
#[repr(C)]
#[majit_macros::jit_immutable_fields("wrappeditems[*]")]
pub struct W_TupleObject {
    pub ob_header: PyObject,
    /// CPython 3.14 `PyTupleObject.ob_hash`; `TUPLE_HASH_UNSET` until computed.
    pub hash: AtomicI64,
    /// `Ptr(GcArray(OBJECTPTR))` — items body, allocation-immutable
    /// per `_immutable_fields_ = ['wrappeditems[*]']`. The GcArray
    /// header `capacity` IS the live tuple length (rlist.py:251
    /// `len(l.items)`); empty tuples carry a 0-cap header-only
    /// allocation (non-null pointer).
    pub wrappeditems: *mut ItemsBlock,
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// The builtin tuple payload stays unchanged; the generated user class adds
/// `MapdictStorageMixin` after it.
#[repr(C)]
pub struct W_TupleObjectUser {
    pub base: W_TupleObject,
    pub map: usize,
    pub storage: *mut ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_TupleObjectUser, storage)
            == std::mem::offset_of!(W_TupleObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// GC type id assigned to `W_TupleObject` at `JitDriver` init time.
/// Held as a constant here (rather than runtime-queried) so
/// pyre-object's host-side allocator can reach it without a
/// back-channel; `pyre-jit/src/eval.rs` asserts the same id is
/// returned by `gc.register_type(...)` so any drift panics on
/// startup. Re-exported from `pyre_jit_trace::descr` for existing
/// call sites.
pub const W_TUPLE_GC_TYPE_ID: u32 = 8;
pub const W_TUPLE_OBJECT_SIZE: usize = std::mem::size_of::<W_TupleObject>();
/// User-subclass tuple layout (`typedef.py` `_getusercls`). Unconditional id 160.
pub const W_TUPLE_USER_GC_TYPE_ID: u32 = 160;
pub const W_TUPLE_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_TupleObjectUser>();

impl crate::lltype::GcType for W_TupleObject {
    fn type_id() -> u32 {
        W_TUPLE_GC_TYPE_ID
    }
    const SIZE: usize = W_TUPLE_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_TupleObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_TUPLE_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_TUPLE_USER_OBJECT_SIZE;
}

// JIT field descriptors read/write the atomic cache as a signed machine word.
const _: () = {
    assert!(std::mem::size_of::<AtomicI64>() == std::mem::size_of::<i64>());
    assert!(std::mem::align_of::<AtomicI64>() == std::mem::align_of::<i64>());
};

/// Raw `(ptr, len)` view of a tuple's inline `PyObjectRef` items for GC root
/// walking.  `wrappeditems` is exact-size (`capacity == len`, every slot
/// written by `alloc_tuple_items_block`), so the block capacity is the live
/// length.  Returns `None` for a null block.  Alloc-free — mirrors
/// `listobject::w_list_object_items_ptr_len` and the stationary branch of
/// `tuple_object_custom_trace`, reading `items_block_items_base` /
/// `items_block_capacity` over the block.  The returned pointer aliases the
/// live backing store; the caller must not mutate the tuple while reading
/// through it.
///
/// # Safety
/// `obj` must point to a valid `W_TupleObject`.
pub unsafe fn w_tuple_object_items_ptr_len(
    obj: PyObjectRef,
) -> Option<(*const PyObjectRef, usize)> {
    let tuple = &*(obj as *const W_TupleObject);
    let block = tuple.wrappeditems;
    if block.is_null() {
        return None;
    }
    let base = tuple_items_base(block);
    let cap = items_block_capacity(block);
    Some((base as *const PyObjectRef, cap))
}

/// CPython 3.14 `r_object()` tuple construction: fill one slot of an
/// array-backed tuple that was allocated with null elements and published in
/// the marshal reference table before its children were read.
///
/// This is deliberately not a general tuple mutator.  The caller must use it
/// only during initial construction, before the tuple escapes to Python code
/// or has its hash observed.  Run the old-to-young barrier before the store,
/// matching `w_tuple_new_array_backed_impl`, and root both operands across the
/// barrier's possible collection.
///
/// # Safety
/// `obj` must be a general array-backed tuple and `index` must name an
/// uninitialised slot.
pub unsafe fn w_tuple_setitem_initializing(
    obj: PyObjectRef,
    index: usize,
    value: PyObjectRef,
) -> bool {
    match w_tuple_setitem_unchecked(obj, index, value) {
        Some(previous) => {
            debug_assert!(previous.is_null());
            true
        }
        None => false,
    }
}

/// Fill one slot of an array-backed tuple whatever the slot holds now, and
/// answer with what it held -- `PyTuple_SET_ITEM`.
///
/// The one thing that separates this from [`w_tuple_setitem_initializing`] is
/// that the slot need not be empty: a caller may swap a value in and put the
/// old one back, which `ffi_obj.c _ffi_callback_decorator` does around the
/// argument tuple it borrows.  The construction-time contract is the same --
/// the tuple must not have escaped to Python code or had its hash observed --
/// and so is everything owed to the collector.
///
/// # Safety
/// `obj` must be a general array-backed tuple.
pub unsafe fn w_tuple_setitem_unchecked(
    obj: PyObjectRef,
    index: usize,
    value: PyObjectRef,
) -> Option<PyObjectRef> {
    let (items, len) = w_tuple_object_items_ptr_len(obj)?;
    if index >= len {
        return None;
    }
    let previous = *items.add(index);
    // `setarrayitem_gc`: the slot belongs to the `wrappeditems` array, a GC
    // object of its own, so that array is the holder `write_barrier`
    // remembers.  Remembering the tuple leaves an old array's young element
    // off `old_objects_pointing_to_young`.
    let block = (*(obj as *const W_TupleObject)).wrappeditems;
    crate::object_array::items_block_set_ref(block, index, value);
    Some(previous)
}

/// Walk, in place, every element `PyObjectRef` slot of any tuple — general
/// `W_TupleObject` or a specialised arity-2 variant — for GC root forwarding.
/// `is_tuple` reports all four layouts as `tuple`, but only the general variant
/// carries a `wrappeditems` block; the specialised `_oo` variant stores its two
/// objects inline (`value0` / `value1`) and `_ii` / `_ff` hold unboxed scalars
/// with no GC children.  Dispatch on the concrete layout exactly as the marker
/// does (`tuple_object_custom_trace` and the `spec_tuple_*` gc_ptr_offsets), so
/// a specialised tuple is never mis-read as a general one.  Alloc-free.
///
/// # Safety
/// `obj` must point to a valid tuple (`is_tuple(obj)` true).
pub unsafe fn w_tuple_walk_gc_refs(obj: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
    use crate::specialisedtupleobject::{
        W_SpecialisedTupleObject_oo, is_specialised_tuple_ff, is_specialised_tuple_ii,
        is_specialised_tuple_oo,
    };
    if is_specialised_tuple_oo(obj) {
        let t = obj as *mut W_SpecialisedTupleObject_oo;
        visitor(std::ptr::addr_of_mut!((*t).value0));
        visitor(std::ptr::addr_of_mut!((*t).value1));
        return;
    }
    // `_ii` / `_ff` are GC-leaf (unboxed scalars) — nothing to forward.
    if is_specialised_tuple_ii(obj) || is_specialised_tuple_ff(obj) {
        return;
    }
    // General `W_TupleObject`: forward each slot of the exact-size items block.
    if let Some((ptr, n)) = w_tuple_object_items_ptr_len(obj) {
        let mut i = 0usize;
        while i < n {
            visitor(ptr.add(i) as *mut PyObjectRef);
            i += 1;
        }
    }
}

/// Allocate a new tuple from a Vec of items.
///
/// Arity-2 tuples are routed through `makespecialisedtuple2`
/// (`pypy/objspace/std/specialisedtupleobject.py`), except that
/// Python 3.14's pointer identity requires exact float references to remain
/// boxed. Other arities use the array-backed `W_TupleObject`.
/// Empty tuples allocate like any other arity; `W_AbstractTupleObject.is_w`
/// / `immutable_unique_id` give `() is ()` without a process-global object.
///
/// Residualized: tuple construction drives the moving collector through
/// `push_roots` / `pin_root` / `try_gc_alloc_nursery_raw` shadow-stack
/// plumbing the tracer cannot model. The JIT leaves the call as a
/// residual returning the fresh object pointer.
#[majit_macros::dont_look_inside]
pub fn w_tuple_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    w_tuple_new_from_slice(&items)
}

/// Slice-taking core of [`w_tuple_new`]. Residualized for the same
/// moving-collector shadow-stack plumbing the tracer cannot model.
#[majit_macros::dont_look_inside]
pub fn w_tuple_new_from_slice(items: &[PyObjectRef]) -> PyObjectRef {
    if items.len() == 2 {
        return wraptuple2(items[0], items[1]);
    }
    w_tuple_new_array_backed_impl(items, get_instantiate(&TUPLE_TYPE), false)
}

/// `W_TupleObject.__init__` when `wrappeditems` is already
/// `make_sure_not_resized`. The header is new; the array is not copied.
///
/// `space.newtuple` shares that list for every length other than 2.
/// Length 2 goes through `makespecialisedtuple` instead.
///
/// # Safety
/// `block` must be a live exact-size items array (`ll_fixed_newlist`).
#[majit_macros::dont_look_inside]
pub unsafe fn w_tuple_adopt_fixed_items(block: *mut ItemsBlock) -> PyObjectRef {
    debug_assert!(!block.is_null());
    let _roots = crate::gc_roots::push_roots();
    let block_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(block as PyObjectRef);
    // `get_instantiate` can allocate the class. The array is already pinned.
    let w_class = get_instantiate(&TUPLE_TYPE);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let ob_type: *const PyType = &TUPLE_TYPE;
    let header = || PyObject {
        ob_type,
        w_class: crate::gc_roots::shadow_stack_get(class_slot),
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_TUPLE_GC_TYPE_ID, W_TUPLE_OBJECT_SIZE);
    if !raw.is_null() {
        let header_now = header();
        unsafe {
            write_tuple_layout(
                raw,
                header_now.ob_type,
                header_now.w_class,
                std::ptr::null_mut(),
                false,
            );
        }
        // Publish the header before any later safepoint. The trace hook
        // returns early while `wrappeditems` is null.
        let _ = crate::gc_roots::pin_root(raw as PyObjectRef);
        let raw_slot = crate::gc_roots::shadow_stack_len() - 1;
        let raw = crate::gc_roots::shadow_stack_get(raw_slot) as crate::gc_hook::GCREF;
        let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
        // Barrier, then the field. The barrier is not a collection point.
        crate::gc_hook::try_gc_write_barrier_managed(raw as crate::gc_hook::GCREF);
        let header_now = header();
        unsafe {
            write_tuple_layout(raw, header_now.ob_type, header_now.w_class, block, false);
        }
        return raw as PyObjectRef;
    }
    let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
    Box::into_raw(Box::new(W_TupleObject {
        ob_header: header(),
        hash: AtomicI64::new(TUPLE_HASH_UNSET),
        wrappeditems: block,
    })) as PyObjectRef
}

/// `tupleobject.py wraptuple`. Length 2 is `wraptuple2`. Every other
/// length stores this exact-size array on a new `W_TupleObject`
/// (`w_tuple_adopt_fixed_items`).
///
/// The array is not freed. Length 2 copies the two references out and
/// leaves the block with the caller; a shared exception `args_w` stays
/// owned by the exception.
///
/// # Safety
/// `block` must be a live exact-size items array (`ll_fixed_newlist`).
pub unsafe fn wraptuple(block: *mut ItemsBlock) -> PyObjectRef {
    debug_assert!(!block.is_null());
    let len = items_block_capacity(block);
    if len == 2 {
        let base = tuple_items_base(block);
        let first = *base;
        let second = *base.add(1);
        return wraptuple2(first, second);
    }
    w_tuple_adopt_fixed_items(block)
}

/// tupleobject.py `wraptuple2`, the body of `space.newtuple2`.
pub fn wraptuple2(w_a: PyObjectRef, w_b: PyObjectRef) -> PyObjectRef {
    // PyPy can use `_ff` here because its object space gives plain floats
    // value identity.  Pyre follows Python 3.14 pointer identity: `(x, x)`
    // must contain the exact `x` object, not two freshly boxed copies.
    //
    // This interception is also the ONLY reason the `_ff` layout has no
    // producer.  `makespecialisedtuple2` below still builds one for a
    // plain-float pair, and this is its sole non-test caller, so deleting
    // the branch to restore the upstream shape does not merely change a
    // tuple's layout — it makes the walker's `ff` specialisation arm live,
    // which documents itself as unreachable.  Retire them together.
    if unsafe { is_plain_float_strict(w_a) && is_plain_float_strict(w_b) } {
        return w_specialised_tuple_oo_new(w_a, w_b);
    }
    makespecialisedtuple2(w_a, w_b)
}

/// Word-ABI residual of a 1-tuple.
///
/// `w_tuple_new(vec![item])` builds a `Vec`, which the walk cannot lower.
/// One `PyObjectRef` argument is a single residual slot.
#[majit_macros::dont_look_inside]
pub fn jit_w_tuple1(item: PyObjectRef) -> PyObjectRef {
    w_tuple_new(vec![item])
}

/// Word-ABI residual of a 2-tuple.
///
/// `w_tuple_new(vec![a, b])` builds a `Vec`, which the walk cannot lower.
/// Two `PyObjectRef` arguments are one residual slot each.
#[majit_macros::dont_look_inside]
pub fn jit_w_tuple2(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    w_tuple_new(vec![a, b])
}

/// Allocate the array-backed `W_TupleObject` directly, bypassing
/// arity-2 specialisation. Useful for tests and call sites that need
/// the canonical layout.
///
/// `wrappeditems` points at a `std::alloc`'d `ItemsBlock`, which is
/// outside the GC heap, so the elements are NOT reachable through inline
/// `gc_ptr_offsets` (the collector would stop at the non-managed block
/// pointer). The type is registered with `tuple_object_custom_trace`
/// (`eval.rs`) which walks the block, and this constructor write-barriers
/// the old-gen tuple so a minor collection actually runs that hook on a
/// tuple holding young elements.
///
/// Residualized for the same GC-allocator reason as `w_tuple_new`.
#[majit_macros::dont_look_inside]
pub fn w_tuple_new_array_backed(items: Vec<PyObjectRef>) -> PyObjectRef {
    w_tuple_new_array_backed_impl(&items, get_instantiate(&TUPLE_TYPE), false)
}

/// Build the array-backed layout used by a tuple user subclass. This is the
/// allocation half of `typedef.py _getusercls`: the base tuple fields keep their
/// offsets and the generated user class alone receives mapdict storage.
#[majit_macros::dont_look_inside]
pub fn w_tuple_subclass_new_array_backed(
    items: Vec<PyObjectRef>,
    w_class: PyObjectRef,
) -> PyObjectRef {
    w_tuple_new_array_backed_impl(&items, w_class, true)
}

fn w_tuple_new_array_backed_impl(
    items: &[PyObjectRef],
    w_class: PyObjectRef,
    user_layout: bool,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`):
    //   livevars = self.push_roots(hop)
    //   v_alloc = hop.genop("direct_call", [malloc_fast_ptr, ...])
    //   self.pop_roots(hop, livevars)
    // Each `items[i]` is a live GC pointer that must survive — and be
    // relocated by — the collection that the GC mallocs below may
    // trigger. `pin_root` records each in the shadow stack so the
    // moving collector both keeps it alive and rewrites its slot to the
    // post-relocation address. The `items_block` is filled only AFTER
    // those mallocs, from the relocated shadow-stack slots, mirroring
    // `pop_roots` reading the (possibly-moved) values back: the local
    // `items` slice still holds pre-collection addresses, and the
    // `std::alloc`'d `items_block` is invisible to the collector until
    // `wrappeditems` is set, so filling it before the tuple malloc would
    // leave it pointing at evacuated nursery slots.
    // The livevar set spans two slices (the items, then `w_class` below), so
    // every `publish` runs before the single `normalize`. A per-item
    // `pin_root` would query the GC after the first write, and that query is a
    // safepoint at which the items not yet published are invisible to a
    // foreign collection — the window `publish_roots` is split out to close.
    let _roots = crate::gc_roots::push_roots();
    let len = items.len();
    let save_point = _roots.publish(items);

    // `w_class` is a livevar of this frame for the same reason the items are:
    // every header below is written *after* the mallocs, so the class word this
    // frame was handed can already name an evacuated slot. It joins the same
    // run, and each header is built out of the slot the collector rewrites.
    let class_slot = _roots.publish(std::slice::from_ref(&w_class));
    _roots.normalize(save_point, len + 1);

    // The only allocations that may collect: the type's lazy instantiate
    // map and the tuple header. Both run while every item is pinned.
    // `_getusercls(W_TupleObject)` is its own class: the user layout carries
    // it as the typeptr.
    let ob_type: *const PyType = if user_layout {
        &crate::pyobject::TUPLE_USER_TYPE
    } else {
        &TUPLE_TYPE
    };
    let header = || PyObject {
        ob_type,
        w_class: crate::gc_roots::shadow_stack_get(class_slot),
    };
    let (type_id, object_size) = if user_layout {
        (W_TUPLE_USER_GC_TYPE_ID, W_TUPLE_USER_OBJECT_SIZE)
    } else {
        (W_TUPLE_GC_TYPE_ID, W_TUPLE_OBJECT_SIZE)
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(type_id, object_size);
    // The freshly allocated tuple header is itself a translated livevar across
    // the items-block allocation and the write barrier below, and the nursery
    // allocator's block moves, so publishing it is what keeps the address this
    // frame later writes through correct: every read of it below comes back out
    // of the shadow-stack slot. In a free-threaded run either operation may
    // also park behind a foreign collector.
    //
    // The allocator hands back UNINITIALIZED payload, and once the root is
    // published `tuple_object_custom_trace` may read `ob_header.w_class` and
    // `wrappeditems` off it at any parking point.  Write both slots first, with
    // a null items block — the state the trace hook already returns early on —
    // and install the real block below.
    let raw_slot = if raw.is_null() {
        None
    } else {
        let header = header();
        unsafe {
            write_tuple_layout(
                raw,
                header.ob_type,
                header.w_class,
                std::ptr::null_mut(),
                user_layout,
            );
        }
        let _ = crate::gc_roots::pin_root(raw as PyObjectRef);
        Some(crate::gc_roots::shadow_stack_len() - 1)
    };

    // pop_roots: fill the items block from the item slots, which a
    // collection during the block malloc rewrites in place. A `Vec`
    // snapshot taken here would still hold pre-move addresses across that
    // allocation. Gate off it is the std::alloc block.
    let items_block = unsafe { alloc_tuple_items_block_gc(save_point, len) };
    let raw = raw_slot
        .map(crate::gc_roots::shadow_stack_get)
        .unwrap_or(std::ptr::null_mut()) as crate::gc_hook::GCREF;

    if !raw.is_null() {
        // The element pointers are stored in the off-GC `items_block`, so
        // the implicit write barrier on the tuple struct never fires —
        // register the tuple explicitly so the next minor collection scans it
        // (via the `wrappeditems` custom-trace hook) and relocates any young
        // element. Mirrors the `write_barrier_from_array` an old list/tuple
        // store would emit (incminimark.py). A tuple that spilled to the
        // old generation because the nursery was full is the case that needs
        // it; one still in the nursery carries no `GCFLAG_TRACK_YOUNG_PTRS`,
        // which is the flag `do_write_barrier_managed` tests, so the call
        // costs it a load and nothing else.
        //
        // This runs BEFORE the store, against the null-items image published
        // above: remembering an old object that holds no young pointer yet is
        // the harmless direction. Storing first would leave the young block
        // named by a slot no collection traces until the barrier runs —
        // `remember_young_pointer` clears `GCFLAG_TRACK_YOUNG_PTRS` and queues
        // the tuple only once it does (`remember_young_pointer`), and a minor
        // landing in that window reclaims the block and poisons the nursery
        // under it, leaving `wrappeditems` dangling for good.
        crate::gc_hook::try_gc_write_barrier_managed(raw as crate::gc_hook::GCREF);
        // The header went in before the root was published; only the items
        // block is still outstanding. The store below is the old-to-young
        // edge (`setfield_gc`: barrier, then the field). The barrier is not a
        // collection point, so `raw` and `items_block` are still current.
        let header = header();
        unsafe {
            write_tuple_layout(
                raw,
                header.ob_type,
                header.w_class,
                items_block,
                user_layout,
            );
        }
        return raw as PyObjectRef;
    }
    if user_layout {
        crate::lltype::malloc_typed(W_TupleObjectUser {
            base: W_TupleObject {
                ob_header: header(),
                hash: AtomicI64::new(TUPLE_HASH_UNSET),
                wrappeditems: items_block,
            },
            map: 0,
            storage: std::ptr::null_mut(),
        }) as PyObjectRef
    } else {
        Box::into_raw(Box::new(W_TupleObject {
            ob_header: header(),
            hash: AtomicI64::new(TUPLE_HASH_UNSET),
            wrappeditems: items_block,
        })) as PyObjectRef
    }
}

unsafe fn write_tuple_layout(
    raw: crate::gc_hook::GCREF,
    ob_type: *const PyType,
    w_class: PyObjectRef,
    wrappeditems: *mut ItemsBlock,
    user_layout: bool,
) {
    let header = PyObject { ob_type, w_class };
    if user_layout {
        unsafe {
            std::ptr::write(
                raw as *mut W_TupleObjectUser,
                W_TupleObjectUser {
                    base: W_TupleObject {
                        ob_header: header,
                        hash: AtomicI64::new(TUPLE_HASH_UNSET),
                        wrappeditems,
                    },
                    map: 0,
                    storage: std::ptr::null_mut(),
                },
            )
        };
    } else {
        unsafe {
            std::ptr::write(
                raw as *mut W_TupleObject,
                W_TupleObject {
                    ob_header: header,
                    hash: AtomicI64::new(TUPLE_HASH_UNSET),
                    wrappeditems,
                },
            )
        };
    }
}

/// CPython 3.14 `FT_ATOMIC_LOAD_SSIZE_RELAXED(v->ob_hash)`, generalized over
/// PyPy's three arity-2 specialized tuple layouts.
///
/// # Safety
/// `obj` must point to a valid tuple object.
#[inline]
pub unsafe fn w_tuple_cached_hash(obj: PyObjectRef) -> Option<i64> {
    let hash = if is_specialised_tuple_ii(obj) {
        (*(obj as *const W_SpecialisedTupleObject_ii))
            .hash
            .load(Ordering::Relaxed)
    } else if is_specialised_tuple_ff(obj) {
        (*(obj as *const W_SpecialisedTupleObject_ff))
            .hash
            .load(Ordering::Relaxed)
    } else if is_specialised_tuple_oo(obj) {
        (*(obj as *const W_SpecialisedTupleObject_oo))
            .hash
            .load(Ordering::Relaxed)
    } else {
        (*(obj as *const W_TupleObject))
            .hash
            .load(Ordering::Relaxed)
    };
    (hash != TUPLE_HASH_UNSET).then_some(hash)
}

/// CPython 3.14 `FT_ATOMIC_STORE_SSIZE_RELAXED(v->ob_hash, acc)`, generalized
/// over PyPy's arity-2 specialized tuple layouts.
///
/// # Safety
/// `obj` must point to a valid tuple object and `hash` must not be
/// `TUPLE_HASH_UNSET`.
#[inline]
pub unsafe fn w_tuple_set_cached_hash(obj: PyObjectRef, hash: i64) {
    debug_assert_ne!(hash, TUPLE_HASH_UNSET);
    if is_specialised_tuple_ii(obj) {
        (*(obj as *const W_SpecialisedTupleObject_ii))
            .hash
            .store(hash, Ordering::Relaxed);
    } else if is_specialised_tuple_ff(obj) {
        (*(obj as *const W_SpecialisedTupleObject_ff))
            .hash
            .store(hash, Ordering::Relaxed);
    } else if is_specialised_tuple_oo(obj) {
        (*(obj as *const W_SpecialisedTupleObject_oo))
            .hash
            .store(hash, Ordering::Relaxed);
    } else {
        (*(obj as *const W_TupleObject))
            .hash
            .store(hash, Ordering::Relaxed);
    }
}

/// `pypy/objspace/std/specialisedtupleobject.py makespecialisedtuple2
/// makespecialisedtuple2`. Picks the most specific variant for two
/// args; falls through to `Cls_oo` when neither operand qualifies for
/// the int-int / float-float fast paths.
///
/// Predicates: `listobject.py is_plain_int1` accepts exact
/// `W_IntObject` (not bool, not int subclass) AND fits-int
/// `W_LongObject`; `type(w) is W_FloatObject` is strict identity. NaNs are
/// excluded because `Cls_ff` would lose their pointer identity when reboxing.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn makespecialisedtuple2(w_arg1: PyObjectRef, w_arg2: PyObjectRef) -> PyObjectRef {
    unsafe {
        if is_plain_int1(w_arg1) && is_plain_int1(w_arg2) {
            return w_specialised_tuple_ii_new(plain_int_w(w_arg1), plain_int_w(w_arg2));
        }
        if is_plain_float_strict(w_arg1)
            && is_plain_float_strict(w_arg2)
            && !w_float_get_value(w_arg1).is_nan()
            && !w_float_get_value(w_arg2).is_nan()
        {
            return w_specialised_tuple_ff_new(
                w_float_get_value(w_arg1),
                w_float_get_value(w_arg2),
            );
        }
        w_specialised_tuple_oo_new(w_arg1, w_arg2)
    }
}

/// `type(w) is W_FloatObject`. Strict identity, no subclass match —
/// `specialisedtupleobject.py` uses `type(w_arg1) is W_FloatObject`
/// directly with no fits-* extension (no `is_plain_float1` helper
/// upstream).
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_plain_float_strict(obj: PyObjectRef) -> bool {
    if !py_type_check(obj, &FLOAT_TYPE) {
        return false;
    }
    let float_typeobj = get_instantiate(&FLOAT_TYPE);
    if float_typeobj.is_null() {
        return (*obj).w_class.is_null();
    }
    let w_class = (*obj).w_class;
    w_class.is_null() || std::ptr::eq(w_class, float_typeobj)
}

/// Get the item at the given index from a tuple — polymorphic over
/// `W_TupleObject` and the three specialised variants.
///
/// Supports negative indexing. Returns None if out of bounds. For
/// `Cls_ii` / `Cls_ff` the unboxed payload is wrapped via
/// `w_int_new` / `w_float_new` (mirrors
/// `specialisedtupleobject.py wraps[i](self.space, value)`).
///
/// # Safety
/// `obj` must point to a valid tuple of any of the four variants.
pub unsafe fn w_tuple_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let ob_type = (*obj).ob_type;
    let index = if std::ptr::eq(ob_type, &TUPLE_TYPE)
        || std::ptr::eq(ob_type, &crate::pyobject::TUPLE_USER_TYPE)
    {
        // `W_TupleObject.getitem`: `self.wrappeditems[index]` under
        // `except IndexError`, i.e. rlist `ll_getitem` with `dum_checkidx`.
        let length = w_tuple_len(obj) as u64;
        let mut index = index as u64;
        if index >= length {
            index = index.wrapping_add(length);
            if index >= length {
                return None;
            }
        }
        index as usize
    } else {
        // `specialisedtupleobject.py getitem`.
        let index = if index < 0 { index + 2 } else { index };
        if index != 0 && index != 1 {
            return None;
        }
        index as usize
    };
    Some(w_tuple_getitem_known(obj, index))
}

/// Internal: read a tuple item at a known-in-bounds index. Splitting
/// this out lets `w_tuple_items_copy_as_vec` reuse the dispatch.
#[inline]
unsafe fn w_tuple_getitem_known(obj: PyObjectRef, idx: usize) -> PyObjectRef {
    let ob_type = (*obj).ob_type;
    if std::ptr::eq(ob_type, &TUPLE_TYPE)
        || std::ptr::eq(ob_type, &crate::pyobject::TUPLE_USER_TYPE)
    {
        let tuple = &*(obj as *const W_TupleObject);
        let base = tuple_items_base(tuple.wrappeditems);
        *base.add(idx)
    } else if std::ptr::eq(ob_type, &SPECIALISED_TUPLE_II_TYPE) {
        let t = &*(obj as *const W_SpecialisedTupleObject_ii);
        match idx {
            0 => w_int_new(t.value0),
            1 => w_int_new(t.value1),
            _ => unreachable!("bounds guard above"),
        }
    } else if std::ptr::eq(ob_type, &SPECIALISED_TUPLE_FF_TYPE) {
        let t = &*(obj as *const W_SpecialisedTupleObject_ff);
        match idx {
            0 => w_float_new(t.value0),
            1 => w_float_new(t.value1),
            _ => unreachable!("bounds guard above"),
        }
    } else {
        debug_assert!(std::ptr::eq(ob_type, &SPECIALISED_TUPLE_OO_TYPE));
        let t = &*(obj as *const W_SpecialisedTupleObject_oo);
        match idx {
            0 => t.value0,
            1 => t.value1,
            _ => unreachable!("bounds guard above"),
        }
    }
}

/// `W_TupleObject.tolist` — `return self.wrappeditems`.
///
/// Arity-2 `Cls_ii` / `Cls_ff` / `Cls_oo` implement `tolist` by allocating
/// a fresh list. They have no `wrappeditems`, so this returns `None` and
/// the caller builds that list.
///
/// # Safety
/// `obj` must point to a valid tuple (`is_tuple`).
pub unsafe fn w_tuple_wrappeditems(obj: PyObjectRef) -> Option<*mut ItemsBlock> {
    let ob_type = unsafe { (*obj).ob_type };
    if !std::ptr::eq(ob_type, &TUPLE_TYPE)
        && !std::ptr::eq(ob_type, &crate::pyobject::TUPLE_USER_TYPE)
    {
        return None;
    }
    let block = unsafe { (*(obj as *const W_TupleObject)).wrappeditems };
    if block.is_null() { None } else { Some(block) }
}

/// Get the length of a tuple — polymorphic over all four variants.
/// `Cls_ii` / `Cls_ff` / `Cls_oo` are arity-2 by construction. The
/// canonical `W_TupleObject` reads `len(wrappeditems)` directly from
/// the GcArray header per upstream `tupleobject.py:376-390` (no
/// inline length cache; `_immutable_fields_ = ['wrappeditems[*]']`).
///
/// # Safety
/// `obj` must point to a valid tuple of any of the four variants.
pub unsafe fn w_tuple_len(obj: PyObjectRef) -> usize {
    // specialisedtupleobject.py's three generated classes each implement
    // `length()` as the constant `typelen`.  Dispatch on the payload vtable
    // directly, as `w_tuple_getitem_known` does below: routing this through
    // `py_type_check` makes the generated walk mix the user-visible canonical
    // tuple class with the three payload layouts and can fall through to a
    // `W_TupleObject.wrappeditems` read on an inline specialised payload.
    let ob_type = unsafe { (*obj).ob_type };
    if std::ptr::eq(ob_type, &SPECIALISED_TUPLE_II_TYPE)
        || std::ptr::eq(ob_type, &SPECIALISED_TUPLE_FF_TYPE)
        || std::ptr::eq(ob_type, &SPECIALISED_TUPLE_OO_TYPE)
    {
        return 2;
    }
    debug_assert!(
        std::ptr::eq(ob_type, &TUPLE_TYPE)
            || std::ptr::eq(ob_type, &crate::pyobject::TUPLE_USER_TYPE)
    );
    let tuple = &*(obj as *const W_TupleObject);
    items_block_capacity(tuple.wrappeditems)
}

/// `tupleobject.py UNROLL_CUTOFF`.
pub const UNROLL_CUTOFF: usize = 10;

/// `W_TupleObject._unroll_condition` —
/// `jit.loop_unrolling_heuristic(self.wrappeditems, self.length(), UNROLL_CUTOFF)`.
/// The specialised variants carry their items inline, so the tuple itself is
/// the `isvirtual` probe for every variant.
pub fn unroll_condition(obj: PyObjectRef) -> bool {
    let len = unsafe { w_tuple_len(obj) };
    majit_rlib::jit::loop_unrolling_heuristic(unsafe { &*obj }, len, UNROLL_CUTOFF)
}

/// Snapshot the tuple's items as an owned `Vec<PyObjectRef>`.
///
/// The returned vector is a plain `Vec` and roots nothing: a caller that
/// allocates while holding it owes the elements pins of its own.
/// Polymorphic over all four variants — `Cls_ii` / `Cls_ff` re-box
/// their inline payloads via `w_int_new` / `w_float_new`.
///
/// `W_TupleObject.getitems_copy` is `wrappeditems[:]`; specialised
/// `getitems_copy` is `tolist` over `unrolling_iterable(range(typelen))`.
/// `_unroll_condition` decides when that copy looks inside.
///
/// # Safety
/// `obj` must point to a valid tuple of any of the four variants.
#[majit_macros::look_inside_iff(unroll_condition)]
pub unsafe fn w_tuple_items_copy_as_vec(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let n = w_tuple_len(obj);
    if std::ptr::eq((*obj).ob_type, &TUPLE_TYPE)
        || std::ptr::eq((*obj).ob_type, &crate::pyobject::TUPLE_USER_TYPE)
    {
        // Fast path: shared backing array, just copy the slice.
        let tuple = &*(obj as *const W_TupleObject);
        let base = tuple_items_base(tuple.wrappeditems);
        let mut out = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            out.push(*base.add(i));
            i += 1;
        }
        return out;
    }
    // The re-boxing path mints one object per element, so element 0 would sit
    // in a plain `Vec` across element 1's allocation.  The specialised tuple
    // itself is also reread for every box and must be reloaded after a move.
    let tuple_roots = crate::gc_roots::push_roots();
    let tuple_slot = tuple_roots.publish(&[obj]);
    tuple_roots.normalize(tuple_slot, 1);
    let mut out = crate::gc_roots::RootedItems::new();
    let mut i = 0usize;
    while i < n {
        out.push(w_tuple_getitem_known(tuple_roots.get(tuple_slot), i));
        i += 1;
    }
    out.take()
}

#[majit_macros::elidable]
pub extern "C" fn jit_tuple_getitem(tuple: PyObjectRef, index: i64) -> PyObjectRef {
    unsafe {
        match w_tuple_getitem(tuple, index) {
            Some(value) => value,
            None => panic!("tuple index out of range in JIT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    /// `typedef.py _getusercls(W_TupleObject)`: a tuple subclass instance
    /// carries `TUPLE_USER_TYPE`, keeps the array-backed payload, and is not
    /// an exact tuple by typeptr.
    #[test]
    fn tuple_subclass_instance_carries_user_typeptr() {
        let w_class = crate::w_type_new("TupleSub", PY_NULL, std::ptr::null_mut());
        let tup = w_tuple_subclass_new_array_backed(vec![w_int_new(4), w_int_new(5)], w_class);
        unsafe {
            assert!(std::ptr::eq(
                (*tup).ob_type,
                &crate::pyobject::TUPLE_USER_TYPE
            ));
            assert!(is_tuple(tup));
            assert!(!crate::pyobject::is_exact_type(tup, &TUPLE_TYPE));
            assert_eq!(w_tuple_len(tup), 2);
            assert_eq!(
                crate::intobject::w_int_get_value(w_tuple_getitem(tup, 1).unwrap()),
                5
            );
            assert_eq!(w_tuple_items_copy_as_vec(tup).len(), 2);
        }
    }

    #[test]
    fn test_tuple_create_and_access() {
        let items = vec![w_int_new(1), w_int_new(2), w_int_new(3)];
        let tup = w_tuple_new(items);
        unsafe {
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 3);
            let item = w_tuple_getitem(tup, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 1);
            let item = w_tuple_getitem(tup, 2).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 3);
        }
    }

    #[test]
    fn test_tuple_negative_index() {
        let items = vec![w_int_new(10), w_int_new(20)];
        let tup = w_tuple_new(items);
        unsafe {
            // Arity-2 int-int specialisation is active.
            assert!(is_specialised_tuple_ii(tup));
            assert!(is_tuple(tup));
            let item = w_tuple_getitem(tup, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 20);
        }
    }

    #[test]
    fn test_tuple_out_of_bounds() {
        let tup = w_tuple_new(vec![w_int_new(1)]);
        unsafe {
            assert!(w_tuple_getitem(tup, 5).is_none());
            assert!(w_tuple_getitem(tup, -5).is_none());
        }
    }

    #[test]
    fn test_jit_tuple_getitem_shares_tuple_semantics() {
        let tup = w_tuple_new(vec![w_int_new(3), w_int_new(5)]);
        unsafe {
            assert_eq!(
                crate::intobject::w_int_get_value(jit_tuple_getitem(tup, 1)),
                5
            );
        }
    }

    #[test]
    fn test_arity2_int_int_routes_to_specialised_ii() {
        let tup = w_tuple_new(vec![w_int_new(7), w_int_new(11)]);
        unsafe {
            assert!(is_specialised_tuple_ii(tup));
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 2);
            let v0 = w_tuple_getitem(tup, 0).unwrap();
            let v1 = w_tuple_getitem(tup, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v0), 7);
            assert_eq!(crate::intobject::w_int_get_value(v1), 11);
        }
    }

    #[test]
    fn test_arity2_float_float_preserves_boxed_identity() {
        let lhs = crate::floatobject::w_float_new(1.5);
        let rhs = crate::floatobject::w_float_new(2.25);
        let tup = w_tuple_new(vec![lhs, rhs]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), lhs);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), rhs);
        }
    }

    #[test]
    fn test_explicit_specialised_float_pair_keeps_pypy_ff_layout() {
        let lhs = crate::floatobject::w_float_new(1.5);
        let rhs = crate::floatobject::w_float_new(2.25);
        let tup = makespecialisedtuple2(lhs, rhs);
        unsafe {
            assert!(is_specialised_tuple_ff(tup));
            assert_eq!(
                crate::floatobject::w_float_get_value(w_tuple_getitem(tup, 0).unwrap()),
                1.5
            );
        }
    }

    /// NaNs must decline the raw-f64 `Cls_ff` specialization.
    #[test]
    fn test_explicit_specialised_nan_pair_declines_ff() {
        let nan = crate::floatobject::w_float_new(f64::NAN);
        let tup = makespecialisedtuple2(nan, nan);
        unsafe {
            assert!(!is_specialised_tuple_ff(tup));
            assert!(is_specialised_tuple_oo(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), nan);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), nan);
        }

        // One NaN slot is enough — both slots are stored raw.
        let finite = crate::floatobject::w_float_new(1.5);
        let mixed = makespecialisedtuple2(finite, nan);
        unsafe {
            assert!(!is_specialised_tuple_ff(mixed));
            assert_eq!(w_tuple_getitem(mixed, 0).unwrap(), finite);
            assert_eq!(w_tuple_getitem(mixed, 1).unwrap(), nan);
        }
    }

    #[test]
    fn unroll_condition_empty_is_true() {
        // `rlib/jit.py loop_unrolling_heuristic`: size == 0 is always true.
        let empty = w_tuple_new(vec![]);
        assert!(unroll_condition(empty));
        let one = w_tuple_new(vec![crate::intobject::w_int_new(1)]);
        // Residual `isconstant` is false, so a non-empty tuple does not unroll.
        assert!(!unroll_condition(one));
    }

    #[test]
    fn test_arity2_nan_pair_preserves_boxed_identity() {
        let nan = crate::floatobject::w_float_new(f64::NAN);
        let tup = w_tuple_new(vec![nan, nan]);
        unsafe {
            assert!(!is_specialised_tuple_ff(tup));
            assert!(is_specialised_tuple_oo(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), nan);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), nan);
        }
    }

    /// `specialisedtupleobject.py:176` checks `type(w) is W_FloatObject`.
    /// A float subclass keeps the W_FloatObject payload shape but has a
    /// different Python-level `w_class`, so it must fall through to `Cls_oo`.
    #[test]
    fn test_arity2_float_subclass_pair_falls_through_to_oo() {
        let lhs = crate::floatobject::w_float_new(1.5);
        let rhs = crate::floatobject::w_float_new(2.25);
        let fake_subclass = &INSTANCE_TYPE as *const PyType as PyObjectRef;
        unsafe {
            (*lhs).w_class = fake_subclass;
            (*rhs).w_class = fake_subclass;
        }

        let tup = w_tuple_new(vec![lhs, rhs]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ff(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), lhs);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), rhs);
        }
    }

    /// `listobject.py:2390` checks `type(w) is W_LongObject`, so a
    /// W_LongObject carrying an app-level int subclass must not take the
    /// int-int specialised tuple path.
    #[test]
    fn test_arity2_long_subclass_pair_falls_through_to_oo() {
        let lhs = crate::longobject::w_long_from_i64(7);
        let rhs = crate::longobject::w_long_from_i64(11);
        let fake_subclass = &INSTANCE_TYPE as *const PyType as PyObjectRef;
        unsafe {
            (*lhs).w_class = fake_subclass;
            (*rhs).w_class = fake_subclass;
        }

        let tup = w_tuple_new(vec![lhs, rhs]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ii(tup));
            assert_eq!(w_tuple_getitem(tup, 0).unwrap(), lhs);
            assert_eq!(w_tuple_getitem(tup, 1).unwrap(), rhs);
        }
    }

    #[test]
    fn test_arity2_mixed_falls_through_to_specialised_oo() {
        let tup = w_tuple_new(vec![w_int_new(7), crate::floatobject::w_float_new(2.0)]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 2);
        }
    }

    #[test]
    fn test_arity_other_uses_array_backing() {
        let tup0 = w_tuple_new(vec![]);
        let tup1 = w_tuple_new(vec![w_int_new(1)]);
        let tup3 = w_tuple_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(is_tuple(tup0));
            assert!(is_tuple(tup1));
            assert!(is_tuple(tup3));
            assert!(!is_specialised_tuple_ii(tup0));
            assert!(!is_specialised_tuple_ii(tup1));
            assert!(!is_specialised_tuple_ii(tup3));
            assert_eq!(w_tuple_len(tup0), 0);
            assert_eq!(w_tuple_len(tup1), 1);
            assert_eq!(w_tuple_len(tup3), 3);
        }
    }

    #[test]
    fn test_copy_as_vec_reboxes_unboxed_values() {
        let tup = w_tuple_new(vec![w_int_new(7), w_int_new(11)]);
        unsafe {
            let items = w_tuple_items_copy_as_vec(tup);
            assert_eq!(items.len(), 2);
            assert_eq!(crate::intobject::w_int_get_value(items[0]), 7);
            assert_eq!(crate::intobject::w_int_get_value(items[1]), 11);
        }
    }

    /// `specialisedtupleobject.py:172-175` `is_plain_int1` accepts both
    /// exact `W_IntObject` and fits-int `W_LongObject`. A tuple of two
    /// fits-int longs must therefore land on `Cls_ii`, with the inline
    /// payload carrying the unwrapped i64 values.
    #[test]
    fn test_arity2_long_long_fits_int_routes_to_specialised_ii() {
        use crate::longobject::w_long_new;
        use majit_rlib::rbigint::RBigInt as BigInt;
        let tup = w_tuple_new(vec![
            w_long_new(BigInt::from(7)),
            w_long_new(BigInt::from(11)),
        ]);
        unsafe {
            assert!(is_specialised_tuple_ii(tup));
            let v0 = w_tuple_getitem(tup, 0).unwrap();
            let v1 = w_tuple_getitem(tup, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v0), 7);
            assert_eq!(crate::intobject::w_int_get_value(v1), 11);
        }
    }

    /// A `W_LongObject` whose value does not fit in a machine int
    /// rejects the int-int specialisation (per `is_plain_int1`'s
    /// `_fits_int()` check) and must fall through to `Cls_oo`.
    #[test]
    fn test_arity2_overflow_long_falls_through_to_oo() {
        use crate::longobject::w_long_new;
        use majit_rlib::rbigint::RBigInt as BigInt;
        let huge = BigInt::from(i64::MAX) * BigInt::from(2);
        let tup = w_tuple_new(vec![w_long_new(huge), w_int_new(0)]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ii(tup));
        }
    }

    /// `bool` is a subclass of `int` but `is_plain_int1` rejects it
    /// (`type(w) is W_IntObject`). A `(True, False)` pair therefore
    /// falls through to `Cls_oo`, not `Cls_ii`.
    #[test]
    fn test_arity2_bool_pair_falls_through_to_oo() {
        use crate::boolobject::w_bool_from;
        let tup = w_tuple_new(vec![w_bool_from(true), w_bool_from(false)]);
        unsafe {
            assert!(is_specialised_tuple_oo(tup));
            assert!(!is_specialised_tuple_ii(tup));
        }
    }
}
