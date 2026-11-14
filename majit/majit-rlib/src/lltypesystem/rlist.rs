//! `rpython/rtyper/lltypesystem/rlist.py` — the low-level body a list is
//! lowered to.
//!
//! Only the scalar body lives here: a `Ptr(GcArray(Float))` /
//! `Ptr(GcArray(Signed))`, which is what `erase([float])` / `erase([int])`
//! produce for the Float / Integer list strategies (`listobject.py`
//! FloatListStrategy / IntegerListStrategy) and what `rbigint._digits` is
//! lowered to. An 8-byte capacity header (the GcArray length, rlist.py:251
//! `len(l.items)`) is followed by inline 8-byte items (f64 or i64). The live
//! list length lives on the enclosing wrapper (rlist.py:116
//! `("length", Signed)`), so the header is the allocated capacity, fixed for
//! the block's lifetime (a grow allocates a fresh block).
//!
//! Items are non-pointer words, so the collector has no inner refs to trace.
//!
//! A fixed list of str is allocated here as well: `HighLevelOp.convertvar`
//! from a rust-heap `Vec<String>` header to an unresized list of str.
//! `FixedSizeListRepr._setup_repr` calls `externalvsinternal(..., gcref=True)`
//! (`rmodel.py`), so the list is `Ptr(GcArray(GCREF))`. That ARRAY is the
//! one gc-pointer items block (`gctypelayout.py get_type_id`): a length
//! word, then pointer items. The host declares the slot
//! ([`gc_object_items_array_gc_type_id`]).

use std::alloc::{Layout, alloc, alloc_zeroed};

/// No host has declared this array's GC type id yet.
///
/// Same value as [`majit_ir::descr::UNSET_GC_TYPE_ID`]. Not `0`:
/// `TypeRegistry::register` hands out `entries.len()`, so `0` is a
/// legitimate slot — whatever the host registers first. A zero sentinel would
/// read "unset" and "the first registered type" the same way, which is the
/// very confusion these ids exist to prevent. `TypeRegistry::register`
/// reserves this value by rejecting an index that cannot fit below it.
///
/// Spelled as a literal: an initializer naming the `majit_ir` const reads
/// as a call to a global outside the extracted LLBC corpus, which the
/// prepass cannot lift.
pub const UNSET_GC_TYPE_ID: u32 = 0xFFFF_FFFF;
const _: () = assert!(UNSET_GC_TYPE_ID == majit_ir::descr::UNSET_GC_TYPE_ID);

/// GC type id for `Ptr(GcArray(Signed))`. This is a distinct ARRAY identity
/// from `GcArray(Float)` even though the collector trace shape is the same —
/// `GcLLDescr_framework.init_array_descr` gives each ARRAY lltype its own tid.
///
/// A type id is not a property of the lltype; it is a **slot in one host's
/// type registry**, handed out by that host's `gc.register_type` in that
/// host's registration order. The collector reads the word back to index the
/// same registry. So a literal baked in here is another host's slot number:
/// any host but the one it was copied from stamps a foreign index into its own
/// GC headers, and the mis-trace surfaces at collection time — a wrong type
/// for a live object, arbitrarily far from the store that wrote it. The host
/// therefore declares its own, the way it already declares the `rbigint`
/// payload's ([`crate::rbigint::set_rbigint_gc_type_id`]).
///
/// One host per process: see [`declare_array_gc_type_id`] for what that costs
/// and where the instance-scoped owner is upstream.
static GC_INT_ARRAY_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(UNSET_GC_TYPE_ID);

/// GC type id for `Ptr(GcArray(Float))` — the [`gc_int_array_gc_type_id`] twin.
static GC_FLOAT_ARRAY_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(UNSET_GC_TYPE_ID);

/// GC type id for the one `GcArray` of gc pointers.
///
/// `externalvsinternal(..., gcref=True)` (`rmodel.py`) stores a `Ptr(STR)`
/// and an instance pointer as `GCREF`, and `gctypelayout.py get_type_id`
/// numbers that ARRAY once. A host declares the slot its registry gave
/// that ARRAY ([`set_gc_object_items_array_gc_type_id`]).
static GC_OBJECT_ITEMS_ARRAY_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(UNSET_GC_TYPE_ID);

/// Publish `id` as the declaring host's slot for `lltype`.
///
/// The cell is process-global because the collector it indexes is: `gc_sync`
/// holds one collector singleton for the process, and the allocator hook these
/// ids feed (`ACTIVE_ALLOC_NURSERY_TYPED`) is a single global cell as well.
/// One host per process is a standing assumption of the whole allocation path,
/// not a shortcut taken here. Upstream carries no such cell:
/// `GcLLDescr_framework.init_array_descr` (`gc.py`) reads the tid off
/// `self.layoutbuilder` and writes it into the individual `ArrayDescr`, both
/// instance-owned, so two backends in one process keep separate ids. Pyre has
/// no descriptor object at the allocation site — the block allocators take a
/// bare `tid: u32` — so there is nothing instance-scoped for the id to live on
/// until that plumbing exists, and moving only these ids off the static
/// would buy no second host while the singleton and the allocator hook remain.
///
/// Re-declaring the SAME slot is how `build_gc` (`pyre-jit::eval`)
/// re-announces its registration when
/// `test_gc_rebuild_verifies_published_subclass_ranges` builds a second
/// collector: the registry belongs to the collector and the registration
/// order is fixed, so that fresh collector hands the same ids back. A
/// DIFFERENT slot is a second host stamping over the first host's
/// registrations, which nothing downstream can detect — the collector reads
/// the word and indexes its own registry with it.
/// Refuse it here in debug rather than let it surface as a mis-trace.
fn declare_array_gc_type_id(slot: &std::sync::atomic::AtomicU32, id: u32, lltype: &str) {
    let previous = slot.swap(id, std::sync::atomic::Ordering::Relaxed);
    debug_assert!(
        previous == UNSET_GC_TYPE_ID || previous == id,
        "{lltype} GC type id re-declared as {id} over {previous}: a second host \
         is stamping its own registry slots over the first host's"
    );
}

/// Declare the slot this host's `gc.register_type` returned for
/// `GcArray(Signed)`. Call it from the same place the host registers the type.
pub fn set_gc_int_array_gc_type_id(id: u32) {
    declare_array_gc_type_id(&GC_INT_ARRAY_GC_TYPE_ID, id, "GcArray(Signed)");
}

/// Declare the slot this host's `gc.register_type` returned for
/// `GcArray(Float)`.
pub fn set_gc_float_array_gc_type_id(id: u32) {
    declare_array_gc_type_id(&GC_FLOAT_ARRAY_GC_TYPE_ID, id, "GcArray(Float)");
}

/// Declare the slot this host's `gc.register_type` returned for
/// `GcArray(GCREF)`. Call it from the same place the host registers the type.
pub fn set_gc_object_items_array_gc_type_id(id: u32) {
    declare_array_gc_type_id(&GC_OBJECT_ITEMS_ARRAY_GC_TYPE_ID, id, "GcArray(GCREF)");
}

/// Return the slots to [`UNSET_GC_TYPE_ID`] so a test can exercise the
/// declaration channel more than once. The production channel is
/// declare-once-per-process; only the tests below need to undo it.
#[cfg(test)]
fn undeclare_array_gc_type_ids() {
    GC_INT_ARRAY_GC_TYPE_ID.store(UNSET_GC_TYPE_ID, std::sync::atomic::Ordering::Relaxed);
    GC_FLOAT_ARRAY_GC_TYPE_ID.store(UNSET_GC_TYPE_ID, std::sync::atomic::Ordering::Relaxed);
    GC_OBJECT_ITEMS_ARRAY_GC_TYPE_ID.store(UNSET_GC_TYPE_ID, std::sync::atomic::Ordering::Relaxed);
}

/// [`UNSET_GC_TYPE_ID`] until a host declares one.
///
/// Undeclared is not defaulted to anything here, and the caller is not asked
/// to check: an items block allocated while the GC owns the heap carries this
/// word in its header, so [`try_alloc_typed_items_block_nursery`] refuses an
/// undeclared id there and says so. Returning the sentinel rather than
/// panicking on the read keeps the `std::alloc` path this file already
/// documents — bare unit tests and the pre-`init_gc_subsystem` bootstrap,
/// where no collector will ever read the word — working without a
/// registration it has no registry for.
#[majit_macros::dont_look_inside]
pub fn gc_int_array_gc_type_id() -> u32 {
    GC_INT_ARRAY_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`gc_int_array_gc_type_id`]'s twin; same undeclared contract.
#[majit_macros::dont_look_inside]
pub fn gc_float_array_gc_type_id() -> u32 {
    GC_FLOAT_ARRAY_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`UNSET_GC_TYPE_ID`] until a host declares the one `GcArray` of gc
/// pointers. Same undeclared contract as [`gc_int_array_gc_type_id`].
#[majit_macros::dont_look_inside]
pub fn gc_object_items_array_gc_type_id() -> u32 {
    GC_OBJECT_ITEMS_ARRAY_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Route items-block allocations through the moving nursery instead of
/// `std::alloc`. Read once; default ON — the nursery path mirrors the
/// `GcArray` body upstream allocates (rlist.py:84) and is validated identical
/// to the `std::alloc` fallback (check.py 158 both backends, both gate states;
/// fannkuch/nbody/spectral_norm timings unchanged). `MAJIT_GC_ITEMSBLOCK=0`
/// (or `off`/`false`) restores the `std::alloc` fallback to bisect a
/// suspected block-GC regression.
///
/// Reads (and lazily initialises) the runtime-mutable `ENABLED` `OnceLock`,
/// not a build-time constant, so the JIT residualizes the call instead of
/// tracing into it. The `-> bool` return fits a single word and it cannot
/// raise.
///
/// This gates every items block, including the `GcArray(OBJECTPTR)` one that
/// lives with the object model — that block is the same lowering of the same
/// upstream file, so one gate answers for both.
#[majit_macros::dont_look_inside]
pub fn itemsblock_gc_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MAJIT_GC_ITEMSBLOCK")
            .map(|v| !matches!(v.trim(), "0" | "off" | "false" | ""))
            .unwrap_or(true)
    })
}

/// `#[repr(C)] { capacity, items: [u64; 0] }` — the `GcArray(Float)` /
/// `GcArray(Signed)` body. Layout: offset 0 = `capacity` (GcArray length
/// header), offset 8 = items[0..capacity]. Items are 8-byte words read as
/// `f64` / `i64` by the wrapper; the JIT-visible array descriptor carries the
/// element type.
#[repr(C)]
#[majit_macros::jit_immutable_fields("capacity")]
pub struct TypedItemsBlock {
    /// Allocated capacity — the GcArray length header (rlist.py:251).
    pub capacity: usize,
    /// Items inline after the header; size known only at allocation time.
    items: [u64; 0],
}

/// MiniMark-allocated by `try_alloc_typed_items_block_nursery` with the tid from
/// `gc_int_array_gc_type_id` / `gc_float_array_gc_type_id`; this impl declares the gckind
/// only (allocation keeps that tid).
impl majit_gc::GcType for TypedItemsBlock {
    fn type_id() -> u32 {
        u32::MAX
    }
    const SIZE: usize = 0;
}

pub const TYPED_ITEMS_BLOCK_ITEMS_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, items);

/// Offset of the `capacity` header the collector reads as the GcArray length.
pub const TYPED_ITEMS_BLOCK_LEN_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, capacity);

/// Length word of that `GcArray` (`rlist.py` fixed-list `ll_length` /
/// `getarraysize`). Offset 0.
pub const OBJECT_ITEMS_LEN_OFFSET: usize = 0;

/// `items[0]` — `get_array_token` `basesize`. One pointer word after the
/// length.
pub const OBJECT_ITEMS_OFFSET: usize = std::mem::size_of::<usize>();

/// `sizeof(ARRAY.OF)` for a gc pointer item.
pub const OBJECT_ITEMS_ITEM_SIZE: usize = std::mem::size_of::<*const ()>();

const _: () = {
    assert!(OBJECT_ITEMS_LEN_OFFSET == 0);
    assert!(OBJECT_ITEMS_OFFSET == std::mem::size_of::<usize>());
    assert!(OBJECT_ITEMS_ITEM_SIZE == std::mem::size_of::<*const ()>());
};

/// Items base pointer (`&items[0]`) of a `TypedItemsBlock`. Null-safe.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn typed_items_block_items_base(block: *mut TypedItemsBlock) -> *mut u8 {
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET) }
}

/// Allocated capacity (GcArray length header) of a `TypedItemsBlock`. 0 for null.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn typed_items_block_capacity(block: *mut TypedItemsBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

pub fn typed_items_block_layout(cap: usize) -> Layout {
    try_typed_items_block_layout(cap).expect("TypedItemsBlock layout")
}

pub fn try_typed_items_block_layout(cap: usize) -> Option<Layout> {
    let items_size = cap.checked_mul(std::mem::size_of::<u64>())?;
    let total = TYPED_ITEMS_BLOCK_ITEMS_OFFSET.checked_add(items_size)?;
    Layout::from_size_align(total, std::mem::align_of::<TypedItemsBlock>()).ok()
}

/// Allocate a scalar GcArray via `malloc_fast` / `collect_and_reserve`.
///
/// This is the `rbigint._digits` allocation shape. Live unboxed `RBigInt`
/// handles across this call must sit in `RBigIntGcRoot`, matching the
/// translator's `_digits` shadow-stack slots. Once a backend owns the heap,
/// allocation failure must remain a failure: a raw fallback would leave
/// `RBigInt._digits` pointing outside the managed heap even though its
/// descriptor traces that field as `GcArray(Signed)`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_typed_items_block_nursery(cap: usize, tid: u32) -> *mut TypedItemsBlock {
    match unsafe { try_alloc_typed_items_block_nursery(cap, tid) } {
        Some(block) => block,
        None if majit_gc::gc_alloc_failed_pending() => std::ptr::null_mut(),
        None => std::alloc::handle_alloc_error(Layout::new::<TypedItemsBlock>()),
    }
}

/// Fallible companion of [`alloc_typed_items_block_nursery`].
///
/// `ll_newlist` (rlist.py) allocates the items array and writes the
/// length header; the items keep whatever the nursery held, because
/// `malloc_zero_filled` is false for incminimark (incminimark.py). Callers
/// that need `[0] * n` clear it themselves with [`typed_items_block_clear`],
/// which is what `ll_alloc_and_set` does (rtyper/rlist.py); callers
/// building a list display write every slot.
///
/// # Safety
/// `tid` must name a registered array type with no destructor and no weakref
/// flag — the `GcArray(Signed)` / `GcArray(Float)` bodies this serves. Every
/// item the caller reads must be one it has written.
pub unsafe fn try_alloc_typed_items_block_nursery(
    cap: usize,
    tid: u32,
) -> Option<*mut TypedItemsBlock> {
    let cap = cap.max(1);
    let layout = try_typed_items_block_layout(cap)?;
    // Nothing owns the heap during a bare unit test or the pre-`init_gc_subsystem`
    // bootstrap, and there the `std::alloc` path below is the whole heap. Once a
    // backend is installed a null is the GC refusing, and the caller must see
    // that rather than a block outside the managed heap
    // (`majit_gc::gc_allocator_installed`).
    if itemsblock_gc_enabled() && majit_gc::gc_allocator_installed() {
        // Past this point the id is written into the block's GC header and the
        // collector will index its type registry with it, so an undeclared one
        // cannot be defaulted: any value picked here would name some other
        // type in this host's registry and mis-trace a live object at the next
        // collection. Fail at the store instead, where the host that skipped
        // the registration is still on the stack.
        assert!(
            tid != UNSET_GC_TYPE_ID,
            "items block allocated with an undeclared GC type id: a type id is \
             a slot in this host's type registry, so the host must pass the \
             value its own `gc.register_type` returned — for the \
             `GcArray(Signed)` / `GcArray(Float)` bodies, via \
             `set_gc_int_array_gc_type_id` / `set_gc_float_array_gc_type_id`"
        );
        // `GcArray(Signed)` / `GcArray(Float)` bodies: no finalizer, not a
        // WEAKREF, so `gct_fv_gc_malloc` (`framework.py`) reaches
        // `malloc_fast`. That is a copy of `malloc_fixedsize` whose
        // nursery bump reaches `collect_and_reserve` on overflow.
        // `malloc_fast(typeid, size)` takes no root: the livevars belong to
        // the caller's `gc_push_roots`, so the root span here is empty. Live
        // unboxed `RBigInt` handles must sit in `RBigIntGcRoot` across this
        // call, the same way the translator puts `_digits` on the shadow
        // stack.
        let mut needs_write_barrier = false;
        let raw = unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_roots(
                tid,
                layout.size(),
                std::ptr::NonNull::<majit_ir::GcRef>::dangling().as_ptr(),
                0,
                &mut needs_write_barrier,
            )
        }
        .0 as *mut u8;
        if raw.is_null() {
            return None;
        }
        let block = raw as *mut TypedItemsBlock;
        unsafe { (*block).capacity = cap };
        return Some(block);
    }
    unsafe {
        let raw = alloc(layout);
        if raw.is_null() {
            return None;
        }
        let block = raw as *mut TypedItemsBlock;
        (*block).capacity = cap;
        Some(block)
    }
}

/// Collecting `malloc_fast` for an items block with the translated graph's
/// complete native live-root span.  This is the Rust ABI of
/// `gct_fv_gc_malloc`'s `push_roots(hop)` / `pop_roots(hop)` bracket; MiniMark
/// consults the span only when the nursery bump reaches
/// `collect_and_reserve`.
///
/// # Safety
/// Every entry in `roots` must be a valid GC object pointer or null.  The
/// slice must remain address-stable for this call and callers must reload its
/// entries after return before dereferencing any pre-call pointer copies.
pub unsafe fn try_alloc_typed_items_block_nursery_rooted(
    cap: usize,
    tid: u32,
    roots: &mut [majit_ir::GcRef],
) -> Option<*mut TypedItemsBlock> {
    let cap = cap.max(1);
    let layout = try_typed_items_block_layout(cap)?;
    if itemsblock_gc_enabled() && majit_gc::gc_allocator_installed() {
        assert!(
            tid != UNSET_GC_TYPE_ID,
            "items block allocated with an undeclared GC type id"
        );
        let mut needs_write_barrier = false;
        let raw = unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_roots(
                tid,
                layout.size(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut needs_write_barrier,
            )
        }
        .0 as *mut u8;
        if raw.is_null() {
            return None;
        }
        let block = raw as *mut TypedItemsBlock;
        unsafe { (*block).capacity = cap };
        return Some(block);
    }
    unsafe { try_alloc_typed_items_block_nursery(cap, tid) }
}

/// `rgc.ll_arrayclear(l.ll_items())` — zero every item of a freshly allocated
/// block, the second half of `ll_alloc_and_set(LIST, count, 0)`
/// (rtyper/rlist.py).
///
/// # Safety
/// `block` must be a live items block.
#[inline]
pub unsafe fn typed_items_block_clear(block: *mut TypedItemsBlock) {
    unsafe {
        std::ptr::write_bytes(
            typed_items_block_items_base(block),
            0,
            (*block).capacity * std::mem::size_of::<u64>(),
        );
    }
}

/// The `[Digit]` list body `rbigint._digits` is lowered to — a
/// `Ptr(GcArray(Signed))`.
///
/// This is a namespace, not a wrapper: the field stays a raw
/// `*mut TypedItemsBlock` so the annotator keeps seeing the source-level
/// `GcArray(Signed)` shape and `RBigInt::digit` keeps projecting `_digits`
/// directly.
///
/// What it owns is the *vocabulary*. Upstream draws this line already:
/// `rbigint.py` writes `[NULLDIGIT] * size` and `rbigint(digits, sign, size)`
/// and never names a malloc, an allocation tier, or an array type id — the
/// rtyper supplies those, `ll_newlist` (rlist.py) and
/// `ll_alloc_and_set` (rtyper/rlist.py). A consumer of this type
/// spells the list operation; the array's GC type id and the tier it is
/// allocated in stay here.
pub struct Digits;

impl Digits {
    /// `ll_newlist(LIST, length)` (rlist.py) — allocate the items
    /// array without initializing it. `malloc_zero_filled` is false for
    /// incminimark (incminimark.py:211), so the items keep whatever the
    /// nursery held.
    ///
    /// # Safety
    /// Every item the caller reads must be one it has written. Use
    /// [`Digits::alloc_and_set_zero`] for the `[NULLDIGIT] * n` shape.
    #[inline]
    #[expect(
        clippy::new_ret_no_self,
        reason = "This is the direct ll_newlist allocator port from rlist.py; the translated low-level result is a TypedItemsBlock pointer rather than a Rust Digits namespace value"
    )]
    pub unsafe fn new(length: usize) -> *mut TypedItemsBlock {
        unsafe { alloc_typed_items_block_nursery(length, gc_int_array_gc_type_id()) }
    }

    /// Fallible [`Digits::new`], for the upstream paths whose translated
    /// allocation carries an explicit `MemoryError` edge.
    ///
    /// # Safety
    /// Same obligation as [`Digits::new`].
    #[inline]
    pub unsafe fn try_new(length: usize) -> Option<*mut TypedItemsBlock> {
        unsafe { try_alloc_typed_items_block_nursery(length, gc_int_array_gc_type_id()) }
    }

    /// `ll_alloc_and_set(LIST, count, 0)` (rtyper/rlist.py) —
    /// `ll_newlist` followed by `rgc.ll_arrayclear`. This is what
    /// `[NULLDIGIT] * count` lowers to.
    ///
    /// # Safety
    /// The result is a fresh block outside any traced owner; the caller must
    /// store it into one before anything can collect.
    #[inline]
    pub unsafe fn alloc_and_set_zero(count: usize) -> *mut TypedItemsBlock {
        unsafe {
            let block = Self::new(count);
            typed_items_block_clear(block);
            block
        }
    }

    /// Fallible [`Digits::alloc_and_set_zero`].
    ///
    /// # Safety
    /// Same obligation as [`Digits::alloc_and_set_zero`].
    #[inline]
    pub unsafe fn try_alloc_and_set_zero(count: usize) -> Option<*mut TypedItemsBlock> {
        unsafe {
            let block = Self::try_new(count)?;
            typed_items_block_clear(block);
            Some(block)
        }
    }

    /// `ll_alloc_and_set(..., 0)` at a malloc whose transformed graph carries
    /// more than one live GC variable.  Kept beside the ordinary list
    /// operation because the root span is allocation machinery, not bigint
    /// arithmetic.
    ///
    /// # Safety
    /// See [`try_alloc_typed_items_block_nursery_rooted`].
    #[inline]
    pub unsafe fn alloc_and_set_zero_rooted(
        count: usize,
        roots: &mut [majit_ir::GcRef],
    ) -> *mut TypedItemsBlock {
        unsafe {
            let block =
                try_alloc_typed_items_block_nursery_rooted(count, gc_int_array_gc_type_id(), roots)
                    .unwrap_or_else(|| {
                        std::alloc::handle_alloc_error(
                            try_typed_items_block_layout(count.max(1))
                                .unwrap_or_else(Layout::new::<TypedItemsBlock>),
                        )
                    });
            // The zeroing stores follow the malloc with no collection point
            // between them, as in `alloc_and_set_zero`; the caller stores the
            // block into a traced owner before anything can collect.
            typed_items_block_clear(block);
            block
        }
    }

    /// The one-element body of a prebuilt list, at process lifetime.
    ///
    /// Upstream's prebuilt `rbigint` digit lists are translated constants with
    /// the same lifetime; see [`alloc_typed_items_block_immortal`].
    ///
    /// # Safety
    /// Only for a prebuilt whose owner is itself immortal.
    #[inline]
    pub unsafe fn prebuilt(item: i64) -> *mut TypedItemsBlock {
        unsafe {
            let block = alloc_typed_items_block_immortal(1);
            *(typed_items_block_items_base(block) as *mut i64) = item;
            block
        }
    }
}

/// Allocate a headerless, process-lifetime `GcArray(Signed/Float)` body.
///
/// This is only for translated prebuilt objects whose module-global owner is
/// itself immortal (not for ordinary arrays). The layout remains the exact
/// `[length][items...]` GcArray shape seen by generated code, while the hybrid
/// collector deliberately treats the address as non-managed. Upstream's
/// prebuilt `rbigint` digit lists have the same process lifetime.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_typed_items_block_immortal(cap: usize) -> *mut TypedItemsBlock {
    let cap = cap.max(1);
    let layout = typed_items_block_layout(cap);
    unsafe {
        let raw = alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let block = raw as *mut TypedItemsBlock;
        (*block).capacity = cap;
        block
    }
}

/// `GcStruct("list", length: Signed, items: Ptr(GcArray(Signed)))` —
/// `ListRepr::from_item_reprs`. MiniMark's GC header sits before the
/// object pointer, so the pointer addresses `length`. Offsets are the
/// `#[repr(C)]` field order the JIT descr uses for that GcStruct.
#[repr(C)]
pub struct GcList {
    pub length: i64,
    pub items: *mut TypedItemsBlock,
}

pub const GC_LIST_LENGTH_OFFSET: usize = std::mem::offset_of!(GcList, length);
pub const GC_LIST_ITEMS_OFFSET: usize = std::mem::offset_of!(GcList, items);

const _: () = assert!(GC_LIST_LENGTH_OFFSET == 0);
const _: () = assert!(GC_LIST_ITEMS_OFFSET == std::mem::size_of::<i64>());
const _: () = assert!(std::mem::size_of::<GcList>() == 2 * std::mem::size_of::<i64>());

/// `rlist.py _ll_list_resize_hint_really` over-allocation with Signed `newsize`: `some = 3 if newsize < 9 else 6; some += newsize >> 3; return newsize + some`.
pub fn rlist_overallocate_signed(newsize: i64) -> i64 {
    let some = if newsize < 9 { 3i64 } else { 6i64 };
    let some = some.wrapping_add(newsize >> 3);
    newsize.wrapping_add(some)
}

/// `_ll_prebuilt_empty_array(GcArray(Signed))` (`rlist.py`, `specialize:memo`):
/// the prebuilt empty items array lives in the binary's data, like the
/// translated constant.
static LL_PREBUILT_EMPTY_SIGNED_ITEMS: TypedItemsBlock = TypedItemsBlock {
    capacity: 0,
    items: [],
};

fn ll_prebuilt_empty_signed_items_block() -> *mut TypedItemsBlock {
    std::ptr::addr_of!(LL_PREBUILT_EMPTY_SIGNED_ITEMS) as *mut TypedItemsBlock
}

/// `_ll_prebuilt_empty_array(GcArray(Float))` (`rlist.py`, `specialize:memo`):
/// the prebuilt empty items array lives in the binary's data, like the
/// translated constant.
static LL_PREBUILT_EMPTY_FLOAT_ITEMS: TypedItemsBlock = TypedItemsBlock {
    capacity: 0,
    items: [],
};

fn ll_prebuilt_empty_float_items_block() -> *mut TypedItemsBlock {
    std::ptr::addr_of!(LL_PREBUILT_EMPTY_FLOAT_ITEMS) as *mut TypedItemsBlock
}

fn clamp_repeat_count(count: i64) -> usize {
    if count < 0 { 0 } else { count as usize }
}

unsafe fn fill_typed_items_block<T: Copy>(block: *mut TypedItemsBlock, count: usize, item: T) {
    unsafe {
        let n = typed_items_block_capacity(block).min(count);
        let base = typed_items_block_items_base(block) as *mut T;
        let mut i = 0usize;
        while i < n {
            *base.add(i) = item;
            i += 1;
        }
    }
}

/// `_ll_list_resize_hint_really` body shared by the 8-byte-item `GcArray`
/// specializations; the copy is a word copy.
unsafe fn ll_list_resize_hint_really_words(
    l: *mut GcList,
    newsize: i64,
    overallocate: bool,
    type_id: u32,
    empty: *mut TypedItemsBlock,
) {
    unsafe {
        if newsize <= 0 {
            debug_assert!(newsize == 0, "negative list length");
            (*l).length = 0;
            majit_gc::gc_write_barrier(majit_ir::GcRef(l as usize));
            (*l).items = empty;
            return;
        }
        let new_allocated = if overallocate {
            rlist_overallocate_signed(newsize)
        } else {
            newsize
        };
        if new_allocated <= 0 {
            // Overflowed overallocation is negative and `malloc` raises.
            majit_gc::raise_memory_error();
            return;
        }
        let mut roots = [majit_ir::GcRef(l as usize)];
        let Some(newitems) =
            try_alloc_typed_items_block_nursery_rooted(new_allocated as usize, type_id, &mut roots)
        else {
            majit_gc::raise_memory_error();
            return;
        };
        let l = roots[0].0 as *mut GcList;
        let before_len = (*l).length;
        if before_len != 0 {
            let p = if before_len < newsize {
                before_len
            } else {
                newsize
            };
            if p > 0 {
                let items = (*l).items;
                std::ptr::copy_nonoverlapping(
                    typed_items_block_items_base(items) as *const i64,
                    typed_items_block_items_base(newitems) as *mut i64,
                    p as usize,
                );
            }
        }
        majit_gc::gc_write_barrier(majit_ir::GcRef(l as usize));
        (*l).items = newitems;
    }
}

/// Residual CondCall target of `_ll_list_resize_ge` (`rlist.py
/// _ll_list_resize_hint_really`) for the Signed GC `ListRepr`.
/// `dont_look_inside` publishes this to `HELPER_FNADDRS`.
#[majit_macros::dont_look_inside]
pub fn _ll_list_resize_hint_really(l: *mut GcList, newsize: i64, overallocate: bool) {
    unsafe {
        ll_list_resize_hint_really_words(
            l,
            newsize,
            overallocate,
            gc_int_array_gc_type_id(),
            ll_prebuilt_empty_signed_items_block(),
        )
    }
}

/// Residual CondCall target of `_ll_list_resize_ge` (`rlist.py
/// _ll_list_resize_hint_really`) for the Float GC `ListRepr`.
/// `dont_look_inside` publishes this to `HELPER_FNADDRS`.
#[majit_macros::dont_look_inside]
pub fn _ll_list_resize_hint_really_f(l: *mut GcList, newsize: i64, overallocate: bool) {
    unsafe {
        ll_list_resize_hint_really_words(
            l,
            newsize,
            overallocate,
            gc_float_array_gc_type_id(),
            ll_prebuilt_empty_float_items_block(),
        )
    }
}

/// Residual `look_inside_iff` trampoline (`rlib/jit.py`) for a fixed
/// `GcArray(Signed)` fill. `dont_look_inside` publishes this to
/// `HELPER_FNADDRS` as the executable `getfunctionptr` target.
#[majit_macros::dont_look_inside]
pub fn _ll_fixed_alloc_and_set_nonnull_i_trampoline(count: i64, item: i64) -> *mut TypedItemsBlock {
    let count = clamp_repeat_count(count);
    unsafe {
        let block = alloc_typed_items_block_nursery(count, gc_int_array_gc_type_id());
        fill_typed_items_block(block, count, item);
        block
    }
}

/// Residual `look_inside_iff` trampoline for a fixed `GcArray(Float)` fill.
#[majit_macros::dont_look_inside]
pub fn _ll_fixed_alloc_and_set_nonnull_f_trampoline(count: i64, item: f64) -> *mut TypedItemsBlock {
    let count = clamp_repeat_count(count);
    unsafe {
        let block = alloc_typed_items_block_nursery(count, gc_float_array_gc_type_id());
        fill_typed_items_block(block, count, item);
        block
    }
}

/// Payload of an unresized list of str: the object-items `GcArray`.
///
/// Length word, then exactly `len` pointer slots (`ll_fixed_length` reads
/// that word; a fixed list has no separate live-length). Each slot is a
/// `GCREF` (`externalvsinternal` of `Ptr(STR)`). The tid is the slot
/// [`gc_object_items_array_gc_type_id`] — the one ARRAY
/// `gctypelayout.py get_type_id` numbers.
///
/// Zero the payload before the caller publishes the length word.
/// `jit_ll_list_str_from_rust_vec` then allocates each `STR` through
/// `mallocstr_rooted`, and `framework.py` `gct_fv_gc_malloc` `push_roots` /
/// `pop_roots` can run a minor collection. IncrementalMiniMark does not
/// zero-fill, so a walk of `items[0..length]` must see NULL until each
/// `STR` is stored (`gc_malloc_array` leaves `GCREF` slots NULL). The
/// payload allocation itself stays non-collecting `alloc_nursery_typed`.
///
/// Payload overflow is `_ll_compute_size` (`mallocHelpers`): MemoryError,
/// then null. A null nursery or immortal block is
/// `_ll_malloc_varsize_no_length`: MemoryError, then null. A collector
/// whose object-items tid is still unset and `len > 0` is `fatalerror`.
/// `get_type_id` assigns that id before any malloc, and an immortal
/// header (`GCFLAG_NO_HEAP_PTRS`) must not hold nursery `STR` pointers.
/// `len == 0` stays immortal. A declared slot is `alloc_nursery_typed`.
/// With no collector the header tid is the declared slot, or `0` while
/// the slot is unset.
fn alloc_list_of_str_payload(len: usize) -> *mut u8 {
    let Some(payload_size) = OBJECT_ITEMS_ITEM_SIZE
        .checked_mul(len)
        .and_then(|items_bytes| OBJECT_ITEMS_OFFSET.checked_add(items_bytes))
    else {
        // `_ll_compute_size`: ovfcheck overflow raises MemoryError.
        majit_gc::raise_memory_error();
        return std::ptr::null_mut();
    };
    let declared = gc_object_items_array_gc_type_id();
    let unset = declared == UNSET_GC_TYPE_ID;
    let collector = majit_gc::gc_allocator_installed();
    if collector && unset && len > 0 {
        // `get_type_id` has already run. Missing the tid is `fatalerror`.
        eprintln!("fatal error: GcArray(GCREF) type id is unset with a collector installed");
        std::process::abort();
    }
    let tid = if unset { 0 } else { declared };
    let raw = if collector && !unset {
        let gcref = majit_gc::alloc_nursery_typed(tid, payload_size);
        if gcref.is_null() {
            std::ptr::null_mut()
        } else {
            gcref.0 as *mut u8
        }
    } else {
        majit_gc::header::alloc_varsize_with_gc_header_immortal(payload_size, tid)
    };
    if raw.is_null() {
        // `_ll_malloc_varsize_no_length`: null from `allocate` raises MemoryError.
        majit_gc::raise_memory_error();
        return std::ptr::null_mut();
    }
    unsafe {
        std::ptr::write_bytes(raw, 0, payload_size);
    }
    raw
}

/// `HighLevelOp.convertvar` from a rust-heap `Vec<String>` (`{cap, ptr, len}`)
/// to an unresized list of str (`FixedSizeListRepr`, `Ptr(GcArray(GCREF))`).
///
/// The source moved the header into the field and elided `Drop`, so this
/// helper consumes it. Each element's bytes are copied the way one `String`
/// is copied into rstr `STR`: `mallocstr_rooted` writes the hash, the
/// length, and the trailing NUL, and the bytes land at `STR_CHARS_OFFSET`.
/// The array is one root across that call (`framework.py` `gct_fv_gc_malloc`
/// `push_roots` / `pop_roots`); the reloaded address is what the next store
/// uses. When a collector is installed, `framework.py`
/// `transform_generic_set` runs `incminimark.py` `write_barrier_from_array`
/// before each GC-pointer store. Each `String` buffer and the `Vec` buffer
/// are then freed. A null source is an empty list. The array length is
/// `len`, including zero. `mallocstr_rooted` returning `MemoryError`
/// (`_ll_malloc_fixedsize` / `_ll_compute_size`) raises and returns null.
/// The partial array is unreachable. The iterator drop frees the
/// `String`s not yet copied and the `Vec` buffer.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_ll_list_str_from_rust_vec(
    src: *mut std::vec::Vec<std::string::String>,
) -> *mut u8 {
    let vec = if src.is_null() {
        std::vec::Vec::new()
    } else {
        unsafe { std::ptr::read(src) }
    };
    let len = vec.len();
    let payload = alloc_list_of_str_payload(len);
    if payload.is_null() {
        drop(vec);
        return std::ptr::null_mut();
    }
    unsafe {
        payload
            .add(OBJECT_ITEMS_LEN_OFFSET)
            .cast::<usize>()
            .write(len);
    }
    // `gct_fv_gc_malloc` `push_roots` / `pop_roots`: this array is the live
    // GC pointer across each string malloc. A minor collection rewrites the
    // slot; every use after the call reads it back.
    let mut roots = [majit_ir::GcRef(payload as usize)];
    let collector = majit_gc::gc_allocator_installed();
    for (index, item) in vec.into_iter().enumerate() {
        let nbytes = item.len();
        let gcref = match unsafe { super::rstr::mallocstr_rooted(nbytes, &mut roots) } {
            Ok(gcref) => gcref,
            Err(crate::rffi::MemoryError) => {
                // `_ll_malloc_fixedsize` / `_ll_compute_size`. The partial
                // array is unreachable. Dropping `item` and returning
                // drops the iterator: the remaining `String`s and the
                // `Vec` buffer.
                majit_gc::raise_memory_error();
                drop(item);
                return std::ptr::null_mut();
            }
        };
        let payload = roots[0].0 as *mut u8;
        let bytes = item.as_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (gcref.0 as *mut u8).add(super::rstr::STR_CHARS_OFFSET),
                bytes.len(),
            );
            if collector {
                // `transform_generic_set` emits `write_barrier_from_array`
                // before a GC-pointer store into an array.
                majit_gc::gc_write_barrier_from_array(majit_ir::GcRef(payload as usize), index);
            }
            payload
                .add(OBJECT_ITEMS_OFFSET)
                .cast::<usize>()
                .add(index)
                .write(gcref.0);
        }
    }
    roots[0].0 as *mut u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The declaration cells are process-global, so the tests that write them
    /// take turns — and each starts from the undeclared state, which is the
    /// only state a fresh process ever presents to a host.
    static DECLARATION: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// Serialize on [`DECLARATION`] and hand back the undeclared state a host
    /// starts from. A test that ends by panicking poisons the lock; the next
    /// test is not the one that failed, so take the guard through the poison.
    ///
    /// This leaves the ids briefly undeclared while other tests in this binary
    /// run. Nothing here installs a GC allocator, and that is the gate
    /// `try_alloc_typed_items_block_nursery` takes its undeclared-id refusal
    /// from, so no sibling can observe the window.
    fn declaration_lock() -> parking_lot::MutexGuard<'static, ()> {
        let guard = DECLARATION.lock();
        undeclare_array_gc_type_ids();
        guard
    }

    /// The declaration channel carries whatever slot the host's registry
    /// handed out, including `0` — which is a real slot (`TypeRegistry::
    /// register` returns `entries.len()`), and which the undeclared state
    /// must stay distinguishable from.
    #[test]
    fn a_declared_array_type_id_reads_back_and_zero_is_a_real_slot() {
        let _guard = declaration_lock();
        set_gc_int_array_gc_type_id(0);
        assert_eq!(gc_int_array_gc_type_id(), 0);
        assert_ne!(gc_int_array_gc_type_id(), UNSET_GC_TYPE_ID);

        set_gc_float_array_gc_type_id(42);
        assert_eq!(gc_float_array_gc_type_id(), 42);
    }

    /// The object-items ARRAY is the third host-declared slot. `0` is a
    /// real registry index (`TypeRegistry::register` returns
    /// `entries.len()`), and the undeclared state must stay distinguishable
    /// from it.
    #[test]
    fn object_items_array_gc_type_id_reads_back() {
        let _guard = declaration_lock();
        set_gc_object_items_array_gc_type_id(0);
        assert_eq!(gc_object_items_array_gc_type_id(), 0);
        assert_ne!(gc_object_items_array_gc_type_id(), UNSET_GC_TYPE_ID);
    }

    /// `build_gc` (`pyre-jit::eval`), called again by
    /// `test_gc_rebuild_verifies_published_subclass_ranges`, re-runs the
    /// host's registrations against a fresh collector, whose registry hands
    /// the same slots back in the same order. That re-declaration is the
    /// supported case, not the overwrite the guard is looking for.
    #[test]
    fn re_declaring_the_same_slot_is_accepted() {
        let _guard = declaration_lock();
        set_gc_int_array_gc_type_id(7);
        set_gc_int_array_gc_type_id(7);
        assert_eq!(gc_int_array_gc_type_id(), 7);
    }

    /// A second host's registry hands out its own slots; stamping them over
    /// the first host's is undetectable once the word reaches a GC header.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic = "re-declared"]
    fn a_conflicting_re_declaration_is_refused() {
        let _guard = declaration_lock();
        set_gc_float_array_gc_type_id(7);
        set_gc_float_array_gc_type_id(8);
    }

    #[test]
    fn i_trampoline_fills_the_block() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let block = _ll_fixed_alloc_and_set_nonnull_i_trampoline(3, 7);
        unsafe {
            assert_eq!(typed_items_block_capacity(block), 3);
            let base = typed_items_block_items_base(block) as *mut i64;
            assert_eq!(*base, 7);
            assert_eq!(*base.add(1), 7);
            assert_eq!(*base.add(2), 7);
        }
    }

    #[test]
    fn f_trampoline_fills_the_block() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let block = _ll_fixed_alloc_and_set_nonnull_f_trampoline(2, 1.5);
        unsafe {
            assert_eq!(typed_items_block_capacity(block), 2);
            let base = typed_items_block_items_base(block) as *mut f64;
            assert_eq!(*base, 1.5);
            assert_eq!(*base.add(1), 1.5);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn residual_alloc_and_set_trampolines_are_published() {
        let module = module_path!().trim_end_matches("::tests");
        let mut published = Vec::new();
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|desc| published.push(desc.path));
        for leaf in [
            "_ll_fixed_alloc_and_set_nonnull_i_trampoline",
            "_ll_fixed_alloc_and_set_nonnull_f_trampoline",
            "_ll_list_resize_hint_really",
            "_ll_list_resize_hint_really_f",
        ] {
            let path = format!("{module}::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }

    #[test]
    fn hint_really_grows_signed_list() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let items = _ll_fixed_alloc_and_set_nonnull_i_trampoline(3, 0);
        unsafe {
            let base = typed_items_block_items_base(items) as *mut i64;
            *base = 10;
            *base.add(1) = 20;
            *base.add(2) = 30;
        }
        let list = Box::into_raw(Box::new(GcList { length: 3, items }));
        _ll_list_resize_hint_really(list, 10, true);
        unsafe {
            assert_eq!((*list).length, 3);
            let grown = (*list).items;
            assert_eq!(
                typed_items_block_capacity(grown),
                rlist_overallocate_signed(10) as usize
            );
            let base = typed_items_block_items_base(grown) as *const i64;
            assert_eq!(*base, 10);
            assert_eq!(*base.add(1), 20);
            assert_eq!(*base.add(2), 30);
            let _ = Box::from_raw(list);
        }
    }

    #[test]
    fn hint_really_grows_float_list() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let items = _ll_fixed_alloc_and_set_nonnull_f_trampoline(3, 0.0);
        unsafe {
            let base = typed_items_block_items_base(items) as *mut f64;
            *base = 10.0;
            *base.add(1) = 20.0;
            *base.add(2) = 30.0;
        }
        let list = Box::into_raw(Box::new(GcList { length: 3, items }));
        _ll_list_resize_hint_really_f(list, 10, true);
        unsafe {
            assert_eq!((*list).length, 3);
            let grown = (*list).items;
            assert!(typed_items_block_capacity(grown) > 3);
            let base = typed_items_block_items_base(grown) as *const f64;
            assert_eq!(*base, 10.0);
            assert_eq!(*base.add(1), 20.0);
            assert_eq!(*base.add(2), 30.0);
            let _ = Box::from_raw(list);
        }
    }

    #[test]
    fn ll_list_resize_hint_really_to_zero_uses_static_empty_block() {
        let mut l = GcList {
            length: 3,
            items: core::ptr::NonNull::<TypedItemsBlock>::dangling().as_ptr(),
        };
        _ll_list_resize_hint_really(&mut l, 0, true);
        assert_eq!(l.length, 0);
        assert_eq!(l.items, ll_prebuilt_empty_signed_items_block());
        _ll_list_resize_hint_really(&mut l, 0, true);
        assert_eq!(l.items, ll_prebuilt_empty_signed_items_block());
    }

    #[test]
    fn hint_really_newsize_zero_resets_to_prebuilt_empty() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let items = _ll_fixed_alloc_and_set_nonnull_i_trampoline(3, 7);
        let list = Box::into_raw(Box::new(GcList { length: 3, items }));
        _ll_list_resize_hint_really(list, 0, true);
        unsafe {
            assert_eq!((*list).length, 0);
            let empty1 = (*list).items;
            assert_eq!(typed_items_block_capacity(empty1), 0);
            _ll_list_resize_hint_really(list, 0, false);
            let empty2 = (*list).items;
            assert_eq!(empty1, empty2);
            let _ = Box::from_raw(list);
        }
    }

    #[test]
    fn ll_list_resize_hint_really_negative_size_raises_memory_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static HITS: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            HITS.fetch_add(1, Ordering::SeqCst);
        }
        let _guard = crate::RAISE_MEMORY_ERROR_HOOK_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _clear = crate::ClearRaiseHook;
        HITS.store(0, Ordering::SeqCst);
        majit_gc::set_active_raise_memory_error(Some(hook));
        let mut l = GcList {
            length: 0,
            items: core::ptr::null_mut(),
        };
        _ll_list_resize_hint_really(&mut l, i64::MAX, true);
        majit_gc::set_active_raise_memory_error(None);
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
        assert!(l.items.is_null());
        assert_eq!(l.length, 0);
    }

    /// `_ll_compute_size` overflow raises MemoryError once and returns null.
    ///
    /// A `Vec<String>` cannot hold this length: `Vec::len` assumes
    /// `len <= MAX_SLICE_LEN`, and `OBJECT_ITEMS_ITEM_SIZE` times that
    /// bound still fits, so the raise is checked on
    /// [`alloc_list_of_str_payload`]. [`jit_ll_list_str_from_rust_vec`]
    /// `drop`s its source `Vec` when that returns null; the `Vec` here
    /// is consumed by the helper and must not crash or leak the
    /// moved-from local.
    #[test]
    fn jit_ll_list_str_from_rust_vec_payload_overflow_raises_memory_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static HITS: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            HITS.fetch_add(1, Ordering::SeqCst);
        }
        let _guard = crate::RAISE_MEMORY_ERROR_HOOK_TEST
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let _clear = crate::ClearRaiseHook;
        HITS.store(0, Ordering::SeqCst);
        majit_gc::set_active_raise_memory_error(Some(hook));

        let mul_overflow = usize::MAX / OBJECT_ITEMS_ITEM_SIZE + 1;
        assert!(OBJECT_ITEMS_ITEM_SIZE.checked_mul(mul_overflow).is_none());
        assert!(alloc_list_of_str_payload(mul_overflow).is_null());
        assert_eq!(HITS.load(Ordering::SeqCst), 1);

        let add_overflow = usize::MAX / OBJECT_ITEMS_ITEM_SIZE;
        let items_bytes = OBJECT_ITEMS_ITEM_SIZE
            .checked_mul(add_overflow)
            .expect("item bytes fit");
        assert!(OBJECT_ITEMS_OFFSET.checked_add(items_bytes).is_none());
        assert!(alloc_list_of_str_payload(add_overflow).is_null());
        assert_eq!(HITS.load(Ordering::SeqCst), 2);

        if majit_gc::gc_allocator_installed() {
            return;
        }
        let _decl = declaration_lock();
        set_gc_object_items_array_gc_type_id(3);
        let mut source = vec!["owned".to_string(), "data".to_string()];
        let payload = jit_ll_list_str_from_rust_vec(&mut source);
        std::mem::forget(source);
        assert!(!payload.is_null());
        assert_eq!(HITS.load(Ordering::SeqCst), 2);
        unsafe { free_list_of_str(payload, &[5, 4]) };
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn jit_ll_list_str_from_rust_vec_is_published() {
        let mut found = false;
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|row| {
            if row
                .path
                .ends_with("lltypesystem::rlist::jit_ll_list_str_from_rust_vec")
            {
                found = true;
                assert_ne!(row.get(), 0);
                assert_eq!(row.arity, 1);
            }
        });
        assert!(
            found,
            "HELPER_FNADDRS must publish jit_ll_list_str_from_rust_vec"
        );
    }

    /// The helper `ptr::read`s the header, so the caller's `Vec` is
    /// moved-from and must not be dropped. `mem::forget` is that contract.
    /// The payload is the object-items `GcArray`: the host-declared tid,
    /// a length word, then pointer items.
    #[test]
    fn jit_ll_list_str_from_rust_vec_copies_bytes_and_consumes_the_vec() {
        if majit_gc::gc_allocator_installed() {
            return;
        }
        let _guard = declaration_lock();
        set_gc_object_items_array_gc_type_id(3);

        let empty = jit_ll_list_str_from_rust_vec(std::ptr::null_mut());
        assert!(!empty.is_null());
        unsafe {
            assert_eq!(list_of_str_len(empty), 0);
            free_list_of_str(empty, &[]);
        }

        let mut spare = std::vec::Vec::<String>::with_capacity(4);
        let spare_ptr = &mut spare as *mut std::vec::Vec<String>;
        let empty_cap = jit_ll_list_str_from_rust_vec(spare_ptr);
        std::mem::forget(spare);
        assert!(!empty_cap.is_null());
        unsafe {
            assert_eq!(list_of_str_len(empty_cap), 0);
            free_list_of_str(empty_cap, &[]);
        }

        let mut items = vec!["a".to_string(), "bb".to_string(), String::new()];
        let items_ptr = &mut items as *mut std::vec::Vec<String>;
        let payload = jit_ll_list_str_from_rust_vec(items_ptr);
        std::mem::forget(items);
        assert!(!payload.is_null());
        let expected = ["a", "bb", ""];
        unsafe {
            let hdr = majit_gc::header::header_of(payload as usize);
            assert_eq!((*hdr).type_id(), gc_object_items_array_gc_type_id());
            assert_eq!(list_of_str_len(payload), expected.len());
            for (index, text) in expected.iter().enumerate() {
                let word = list_of_str_item(payload, index);
                let s = word as *const super::super::rstr::STR;
                assert_eq!((*s).hash, 0);
                assert_eq!((*s).length, text.len());
                assert_eq!(super::super::rstr::str_as_bytes(s), text.as_bytes());
                assert_eq!(
                    *((s as *const u8).add(super::super::rstr::STR_CHARS_OFFSET + text.len())),
                    0
                );
            }
            free_list_of_str(payload, &expected.map(str::len));
        }
    }

    unsafe fn list_of_str_len(payload: *mut u8) -> usize {
        unsafe { *payload.add(OBJECT_ITEMS_LEN_OFFSET).cast::<usize>() }
    }

    unsafe fn list_of_str_item(payload: *mut u8, index: usize) -> usize {
        unsafe { *payload.add(OBJECT_ITEMS_OFFSET).cast::<usize>().add(index) }
    }

    unsafe fn free_list_of_str(payload: *mut u8, item_lens: &[usize]) {
        for (index, nbytes) in item_lens.iter().copied().enumerate() {
            let word = unsafe { list_of_str_item(payload, index) };
            let str_payload = super::super::rstr::STR_CHARS_OFFSET + nbytes + 1;
            unsafe {
                majit_gc::header::dealloc_varsize_with_gc_header(
                    word as majit_gc::header::GCREF,
                    str_payload,
                );
            }
        }
        let array_payload = OBJECT_ITEMS_OFFSET + OBJECT_ITEMS_ITEM_SIZE * item_lens.len();
        unsafe {
            majit_gc::header::dealloc_varsize_with_gc_header(
                payload as majit_gc::header::GCREF,
                array_payload,
            );
        }
    }
}
