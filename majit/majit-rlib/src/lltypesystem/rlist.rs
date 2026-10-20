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
//! The `GcArray(OBJECTPTR)` body (rlist.py:84) is not here: it is reachable
//! only from the object model that owns its item type.

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
/// until that plumbing exists, and moving only these two ids off the static
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

/// Return both slots to [`UNSET_GC_TYPE_ID`] so a test can exercise the
/// declaration channel more than once. The production channel is
/// declare-once-per-process; only the tests below need to undo it.
#[cfg(test)]
fn undeclare_array_gc_type_ids() {
    GC_INT_ARRAY_GC_TYPE_ID.store(UNSET_GC_TYPE_ID, std::sync::atomic::Ordering::Relaxed);
    GC_FLOAT_ARRAY_GC_TYPE_ID.store(UNSET_GC_TYPE_ID, std::sync::atomic::Ordering::Relaxed);
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

pub const TYPED_ITEMS_BLOCK_ITEMS_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, items);

/// Offset of the `capacity` header the collector reads as the GcArray length.
pub const TYPED_ITEMS_BLOCK_LEN_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, capacity);

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
    unsafe {
        try_alloc_typed_items_block_nursery(cap, tid)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(Layout::new::<TypedItemsBlock>()))
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
        ] {
            let path = format!("{module}::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }
}
