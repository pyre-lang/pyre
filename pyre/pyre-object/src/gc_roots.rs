//! `push_roots` / `pop_roots` parity scaffold.
//!
//! Mirrors RPython's `framework.py gct_fv_gc_malloc`, which
//! brackets every GC malloc call with `push_roots(hop)` and
//! `pop_roots(hop, livevars)`:
//!
//! ```text
//! self.push_roots(hop)
//! v_alloc = hop.genop("direct_call", [malloc_fast_ptr,
//!                                     c_const_gc, c_type_id, c_size,
//!                                     c_false, c_false, c_false],
//!                     resulttype=llmemory.GCREF)
//! self.pop_roots(hop, livevars)
//! ```
//!
//! `push_roots` collects the values of every live GC pointer in the
//! caller's frame into shadow-stack slots so the moving collector can
//! see them as roots. `pop_roots` reads the (possibly-relocated)
//! values back out after the allocation may have triggered
//! collection.
//!
//! In Rust the natural shape is RAII: [`push_roots`] returns a
//! [`RootScope`] guard; dropping the guard executes the equivalent of
//! `pop_roots`. Callers wrap an allocation site like this:
//!
//! ```ignore
//! let _roots = pyre_object::gc_roots::push_roots();
//! pyre_object::gc_roots::pin_root(some_pyobject_ref);
//! let p = pyre_object::lltype::malloc_typed(value);
//! // _roots is dropped here; the pin is rewound, mirroring
//! // pop_roots(livevars).
//! ```
//!
//! ## What this scaffold stands in for
//!
//! Upstream writes no bracket by hand. The transformer inserts
//! `push_roots` / `pop_roots` around the operations that can collect,
//! across the whole translated graph, so an author cannot forget one and a
//! reviewer never has to look for it. pyre now performs the same insertion
//! and colouring over translated flowspace graphs. The interpreter source
//! also runs directly as rustc-compiled Rust, where those graph operations do
//! not exist; that native execution path retains hand-written brackets, and
//! their count is the size of the native audit surface. From the repo root,
//! excluding this module:
//!
//! ```text
//! rg -o 'push_roots\(\)'     --glob '!target' --glob '!**/gc_roots.rs' pyre/ majit/ | wc -l   # 1376
//! rg -o 'pin_root\('         --glob '!target' --glob '!**/gc_roots.rs' pyre/ majit/ | wc -l   # 2604
//! rg -o 'shadow_stack_get\(' --glob '!target' --glob '!**/gc_roots.rs' pyre/ majit/ | wc -l   # 3571
//! rg -l 'push_roots\(\)'     --glob '!target' --glob '!**/gc_roots.rs' pyre/ majit/ | wc -l   # 163
//! ```
//!
//! A forgotten bracket is not a compile error, so nothing downstream
//! assumes the set is complete. `crate::gc_interp` allocates interpreter
//! boxes born-old through `try_gc_alloc_stable`, and the safepoint major
//! is `MiniMarkGC::do_collect_oldgen_nonmoving`, which leaves the nursery
//! byte-for-byte intact. Both exist because an unrooted [`PyObjectRef`] on
//! the Rust stack of a bytecode handler would dangle across a moving
//! minor. Translated moving-nursery allocation is covered by the automatic
//! graph pass; the native path remains on the born-old/nonmoving contract and
//! these source brackets.
//!
//! ## State
//!
//! [`push_roots`] snapshots the thread-local shadow-stack length into a
//! [`RootScope`]; [`pin_root`] appends a [`PyObjectRef`]; [`Drop`]
//! truncates back to the saved length (`shadowstack.py walk_stack_root`).
//! [`walk_shadow_stack`] hands the pinned roots to the collector:
//! `pyre-jit::eval` registers a `pyre-object`-to-`majit-gc` adapter
//! through `majit_gc::shadow_stack::register_extra_root_walker`, so a
//! pinned pointer is visible to the active `MiniMarkGC` and is read back
//! forwarded.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::Cell;
use std::marker::PhantomData;

use crate::pyobject::PyObjectRef;

/// `shadowstack.py root_stack_depth` — the slot count `ShadowStackPool`
/// raw-mallocs for a thread's root stack before any growth.  `majit-gc`'s
/// jitframe root stack starts from the same constant.
const DEFAULT_ROOT_STACK_DEPTH: usize = 163840;

/// `gcdata.root_stack_base` / `root_stack_top` (`shadowstack.py:75-88`), with
/// the `limit` cell `majit-gc`'s jitframe root stack also keeps so a push can
/// see it has reached `root_stack_depth` without a separate depth load.
///
/// Deliberately holds no `Vec` and no `Box`.  A growable container puts drop
/// glue on the thread-local, and a thread-local carrying a destructor costs a
/// registration-state check on every access — `pin_root` and
/// `shadow_stack_get` sit on the interpreter's allocation and call paths, so
/// that check is not free: dropping it measures `.8711` / `.8727` on the
/// any/all rooting probes with a pure-integer control at exactly `1.0000`.
/// The buffer is raw-allocated once instead, as `_prepare_unused_stack` does
/// (`shadowstack.py`), and released by [`RootStackOwner`].
pub struct RootStack {
    base: Cell<*mut PyObjectRef>,
    top: Cell<*mut PyObjectRef>,
    limit: Cell<*mut PyObjectRef>,
}

impl RootStack {
    const fn new() -> Self {
        Self {
            base: Cell::new(std::ptr::null_mut()),
            top: Cell::new(std::ptr::null_mut()),
            limit: Cell::new(std::ptr::null_mut()),
        }
    }

    /// Slots currently in use — `root_stack_top - root_stack_base`.
    #[inline(always)]
    fn len(&self) -> usize {
        let base = self.base.get();
        if base.is_null() {
            return 0;
        }
        // SAFETY: `top` is derived from `base` by pointer arithmetic within
        // the one live allocation, so the offset is in range.
        unsafe { self.top.get().offset_from(base) as usize }
    }

    /// Slot count the current buffer can hold — `root_stack_depth`.
    fn capacity(&self) -> usize {
        let base = self.base.get();
        if base.is_null() {
            return 0;
        }
        // SAFETY: `limit` is the one-past-the-end pointer of the allocation
        // `base` starts.
        unsafe { self.limit.get().offset_from(base) as usize }
    }

    fn layout_for(depth: usize) -> Layout {
        Layout::array::<PyObjectRef>(depth).expect("root stack layout")
    }

    /// `ShadowStackPool.increase_root_stack_depth` (`shadowstack.py`):
    /// allocate a larger buffer, copy the used portion over, and re-derive
    /// base/top; the depth can never shrink.  Unlike upstream this also serves
    /// as the initial `_prepare_unused_stack`, so a null base is the
    /// "no buffer yet" case rather than a separate pooled one.
    #[cold]
    fn grow(&self, new_depth: usize) {
        let old_base = self.base.get();
        let used = self.len();
        let new_depth = new_depth.max(DEFAULT_ROOT_STACK_DEPTH).max(used);
        if new_depth <= self.capacity() {
            return; // can't easily decrease the size
        }
        // Registering this thread's free-at-exit hook is what keeps the hot
        // `ROOT_STACK` key destructor-free; touching it here rather than on
        // every push keeps that cost off the push path.
        let _ = ROOT_STACK_OWNER.try_with(|_| ());
        // SAFETY: a non-zero array layout; `alloc_zeroed` gives null slots,
        // which the walkers read as "no root" exactly like a null GCREF.
        let new_base = unsafe { alloc_zeroed(Self::layout_for(new_depth)) } as *mut PyObjectRef;
        assert!(!new_base.is_null(), "root stack allocation failed");
        if !old_base.is_null() {
            // SAFETY: `used` slots are live in the old buffer and the new
            // allocation is at least that large; the two never overlap.
            unsafe { std::ptr::copy_nonoverlapping(old_base, new_base, used) };
            // SAFETY: the old buffer came from this same allocator with the
            // layout its own capacity names, and nothing references it now —
            // callers hold indices, not slot addresses, across a push.
            unsafe { dealloc(old_base as *mut u8, Self::layout_for(self.capacity())) };
        }
        self.base.set(new_base);
        // SAFETY: both offsets are inside (or one past the end of) the new
        // allocation.
        unsafe {
            self.top.set(new_base.add(used));
            self.limit.set(new_base.add(new_depth));
        }
    }

    /// `incr_stack(1)` (`shadowstack.py`) — return the slot that was the
    /// top and advance past it.  Upstream's bump is unchecked because the
    /// native stack check fires first, with `root_stack_depth` scaled to the
    /// recursion limit.  pyre pins one slot per argument as well as per live
    /// variable, so a single variadic call can outrun any fixed headroom; the
    /// limit test grows instead of trusting that margin.
    #[inline(always)]
    fn incr_stack(&self) -> *mut PyObjectRef {
        let top = self.top.get();
        if top == self.limit.get() {
            self.grow(self.capacity() * 2);
            return self.incr_stack();
        }
        // SAFETY: `top` is below `limit`, so it names a live slot and the
        // advanced pointer is at most one past the end.
        self.top.set(unsafe { top.add(1) });
        top
    }

    /// `decr_stack` to an absolute depth — the `pop_roots` rewind.
    #[inline(always)]
    fn truncate(&self, len: usize) {
        if len < self.len() {
            // SAFETY: `len` is below the live length, so the slot is inside
            // the allocation.
            self.top.set(unsafe { self.base.get().add(len) });
        }
    }

    /// Address of slot `index`.
    #[inline(always)]
    fn slot(&self, index: usize) -> *mut PyObjectRef {
        // Debug-only, because upstream reaches a shadow-stack slot by pointer
        // arithmetic off `root_stack_top` with no check at all
        // (shadowstack.py:76-88 stores at a compile-time colour offset).
        // A panic path in the release body is a call, which forces a stack
        // frame onto `normalize_published_slot` and keeps it from inlining
        // into the pin it belongs to.  Every caller claimed `index` from
        // `incr_stack` inside a bracket that is still open.
        debug_assert!(index < self.len(), "shadow-stack index out of range");
        // SAFETY: the claimed index is inside the live buffer.
        unsafe { self.base.get().add(index) }
    }
}

/// Frees [`ROOT_STACK`]'s buffer when the thread exits.  It lives in its own
/// thread-local key, touched only from `RootStack::grow`, so the key on the
/// push/read path keeps no destructor of its own.
struct RootStackOwner;

impl Drop for RootStackOwner {
    fn drop(&mut self) {
        // `ROOT_STACK` has no destructor, so it is never torn down and this
        // access cannot fail; `try_with` keeps that from being load-bearing.
        let _ = ROOT_STACK.try_with(|stack| {
            let base = stack.base.get();
            if base.is_null() {
                return;
            }
            let layout = RootStack::layout_for(stack.capacity());
            stack.base.set(std::ptr::null_mut());
            stack.top.set(std::ptr::null_mut());
            stack.limit.set(std::ptr::null_mut());
            // SAFETY: the buffer came from this allocator with this layout and
            // the cells that named it have just been cleared.
            unsafe { dealloc(base as *mut u8, layout) };
        });
    }
}

thread_local! {
    /// Thread-local shadow stack — pyre's mirror of RPython's
    /// `framework.shadowstack` (`shadowstack.py`). Append-only across
    /// each [`RootScope`]'s lifetime; the matching [`Drop`] rewinds `top`
    /// back to the saved length.
    ///
    /// Access is raw, matching RPython's base/top shadow-stack shape. Debug
    /// builds reject mutation during a root walk through
    /// `SHADOW_STACK_WALK_IN_PROGRESS`; release builds pay no bookkeeping.
    static ROOT_STACK: RootStack = const { RootStack::new() };

    /// Registered on the first `grow`; see [`RootStackOwner`].
    static ROOT_STACK_OWNER: RootStackOwner = const { RootStackOwner };

    /// Whether this thread is currently driving a shadow-stack root walk.
    #[cfg(debug_assertions)]
    static SHADOW_STACK_WALK_IN_PROGRESS: Cell<bool> = const { Cell::new(false) };
}

#[cfg(debug_assertions)]
struct ShadowStackWalk<'a> {
    in_progress: &'a Cell<bool>,
}

#[cfg(debug_assertions)]
impl<'a> ShadowStackWalk<'a> {
    fn new(in_progress: &'a Cell<bool>) -> Self {
        // Arm the flag outside the assertion: the walk must be marked whether
        // or not assertions are compiled in, and an assertion that doubles as
        // the only writer stops arming the moment it is disabled.
        let was_walking = in_progress.replace(true);
        debug_assert!(!was_walking, "a shadow-stack root walk was re-entered");
        Self { in_progress }
    }
}

#[cfg(debug_assertions)]
impl Drop for ShadowStackWalk<'_> {
    fn drop(&mut self) {
        self.in_progress.set(false);
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
fn assert_shadow_stack_not_walking() {
    SHADOW_STACK_WALK_IN_PROGRESS.with(|in_progress| {
        debug_assert!(
            !in_progress.get(),
            "the shadow stack was mutated while a root walk was in progress"
        );
    });
}

/// Run `f` against this thread's root stack.  The borrow guards the old
/// `Vec` shape needed are gone with it: base/top are `Cell`s, so no
/// reference is held across a call that could push.
#[inline(always)]
fn with_shadow_stack<R>(f: impl FnOnce(&RootStack) -> R) -> R {
    ROOT_STACK.with(f)
}

/// Normalize one already-published root in place.
///
/// RPython's caller and callee livevars are one GC-transformed graph, so a
/// collection between argument marshalling and the callee's root push rewrites
/// the value the callee sees.  A native JIT call copies the argument into the
/// host ABI first: its jitframe home is rewritten, but that register/stack copy
/// can still name the forwarding stub when the callee publishes it, so follow
/// nursery forwarding once here.
///
/// One query answers it.  `gc_current_object_address` is a nursery range
/// compare plus, for a nursery address, the `is_forwarded` header test
/// (`incminimark.py`); a forwarding target is always an old-gen
/// address, so asking a second time about the value the first store left can
/// only return it unchanged.  Nor does the slot need re-reading for a foreign
/// mutator: it is *published*, which is exactly what lets a foreign root walk
/// rewrite it in place, the same protocol upstream relies on and the reason
/// its mutator never asks whether a root has moved.
#[inline]
fn normalize_published_slot(stack: &RootStack, index: usize) -> PyObjectRef {
    if let Some((start, end, tagged)) = majit_gc::published_nursery_window()
        && !majit_gc::gc_sync::foreign_mutator_seen()
    {
        // SAFETY: `slot` bounds-checks `index`, which the caller claimed.
        return unsafe { normalize_slot_in_window(stack.slot(index), start, end, tagged) };
    }
    normalize_published_slot_hooked(stack, index)
}

/// Follow same-thread nursery forwarding for one published slot against a
/// window the caller already read.
///
/// # Safety
/// `slot` must name a live, writable shadow-stack slot.
#[inline(always)]
unsafe fn normalize_slot_in_window(
    slot: *mut PyObjectRef,
    start: usize,
    end: usize,
    tagged: bool,
) -> PyObjectRef {
    // SAFETY: the caller owns the slot for the duration of this call.
    let root = unsafe { *slot };
    let current = majit_gc::gc_current_object_address_in_window(root as usize, start, end, tagged)
        as PyObjectRef;
    if current != root {
        // SAFETY: same live slot.
        unsafe { *slot = current };
    }
    current
}

/// Normalize a contiguous run of already-published roots.
///
/// `publish` writes the whole livevar set before any query runs, so one window
/// read describes every slot in the run; the disarmed and free-threaded paths
/// stay per-slot because each of their queries is its own safepoint.
///
/// Answers whether any slot in the run named a forwarding stub, so a caller
/// holding a native copy of the same words knows whether it owes a reload.
#[inline]
fn normalize_published_run(stack: &RootStack, base: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let Some((start, end, tagged)) =
        majit_gc::published_nursery_window().filter(|_| !majit_gc::gc_sync::foreign_mutator_seen())
    else {
        let mut moved = false;
        for index in base..base + len {
            // SAFETY: `publish` claimed every index in this range.
            let before = unsafe { *stack.slot(index) };
            moved |= normalize_published_slot_hooked(stack, index) != before;
        }
        return moved;
    };
    assert!(base + len <= stack.len(), "shadow-stack run out of range");
    // SAFETY: bounds-checked against the live length just above, so every
    // offset below names a slot inside the one live allocation.
    let first = unsafe { stack.base.get().add(base) };
    let mut moved = false;
    for offset in 0..len {
        // SAFETY: `offset` is below `len`, which the assertion covers.
        unsafe {
            let slot = first.add(offset);
            let before = *slot;
            moved |= normalize_slot_in_window(slot, start, end, tagged) != before;
        }
    }
    moved
}

/// [`normalize_published_slot`] through the hook chain — the shape that has to
/// answer when no nursery window is published, or when a foreign mutator makes
/// the query a safepoint.
fn normalize_published_slot_hooked(stack: &RootStack, index: usize) -> PyObjectRef {
    // SAFETY: callers claimed `index` before entering this helper and keep the
    // surrounding RootScope alive throughout it.
    let slot = stack.slot(index);
    let root = unsafe { *slot };
    let current = majit_gc::gc_current_object_address(root as usize) as PyObjectRef;
    if current != root {
        // SAFETY: same live slot.
        unsafe { *slot = current };
    }
    current
}

/// `increase_root_stack_depth(new_depth)` (`rlib/rgc.py` →
/// `shadowstack.py`).  `sys.setrecursionlimit` scales the root stack
/// with the limit at `pypy/module/sys/vm.py`; the depth can only grow.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn increase_root_stack_depth(new_depth: usize) {
    with_shadow_stack(|stack| stack.grow(new_depth));
}

/// `root_stack_depth` — the slot count this thread's root stack can hold.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn root_stack_depth() -> usize {
    with_shadow_stack(|stack| {
        let capacity = stack.capacity();
        if capacity == 0 {
            DEFAULT_ROOT_STACK_DEPTH
        } else {
            capacity
        }
    })
}

/// RAII guard returned by [`push_roots`].
///
/// Holding a `RootScope` corresponds to RPython's `livevars` list
/// living across the bracketed allocation call. Dropping the scope
/// executes the equivalent of `pop_roots(hop, livevars)` by rewinding
/// the shadow stack to the length captured at construction time.
#[must_use = "RootScope is the live-roots guard; dropping it pops the roots — \
              binding it to `_roots` makes the scope explicit, while binding \
              to `_` would pop immediately and defeat the bracket"]
pub struct RootScope {
    /// Length of [`ROOT_STACK`] at the moment [`push_roots`] was
    /// called. Will be replaced with a backend
    /// `RootSet::Handle` once the active GC consumes the stack.
    save_point: usize,
    /// A root bracket belongs to the thread whose shadow stack supplied this
    /// save point. Moving it could truncate an unrelated thread's stack.
    _not_send: PhantomData<*const ()>,
}

impl RootScope {
    /// Internal constructor. Callers should always go through
    /// [`push_roots`] so the API surface stays uniform across phases.
    #[inline]
    fn new() -> Self {
        Self {
            save_point: shadow_stack_len(),
            _not_send: PhantomData,
        }
    }

    /// First slot owned by this root bracket.
    #[inline]
    pub fn base(&self) -> usize {
        self.save_point
    }

    /// Scope-local [`pin_root`] on this thread's root-stack cell.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    #[must_use = "a pinned root may have been normalized; use the returned live word or bind it to `let _ =` for liveness-only pins"]
    pub fn pin_root(&self, root: PyObjectRef) -> PyObjectRef {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        let stack_slot = shadow_stack_cell();
        // SAFETY: `stack_slot` is this thread's root-stack cell, alive for the
        // bracket; `incr_stack` returns the slot it just claimed.
        let index = unsafe {
            let stack = &*stack_slot;
            let index = stack.len();
            *stack.incr_stack() = root;
            index
        };
        // SAFETY: `stack_slot` is this thread's live root-stack cell.
        normalize_published_slot(unsafe { &*stack_slot }, index)
    }

    /// Scope-local [`shadow_stack_get`] on this thread's root-stack cell.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    pub fn get(&self, index: usize) -> PyObjectRef {
        let stack_slot = shadow_stack_cell();
        // SAFETY: same cell; `slot` bounds-checks `index`.
        unsafe { *(*stack_slot).slot(index) }
    }

    /// Scope-local [`publish_roots`] on this thread's root-stack cell.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    pub fn publish(&self, roots: &[PyObjectRef]) -> usize {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        let stack_slot = shadow_stack_cell();
        // SAFETY: this thread's cell, alive for the bracket; `incr_stack`
        // returns the slot it just claimed.
        unsafe {
            let stack = &*stack_slot;
            let base = stack.len();
            for &root in roots {
                *stack.incr_stack() = root;
            }
            base
        }
    }

    /// Scope-local [`pin_roots`]: publish every live word, then normalize
    /// the run. Sequential [`Self::pin_root`] would query after the first
    /// write and leave later values invisible to a foreign collection.
    ///
    /// One cell resolution covers both halves: `publish` then
    /// `normalize` each used to call [`shadow_stack_cell`], and the
    /// second `_tlv_get_addr` sat on every `with_roots!` bracket.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    pub fn pin_roots(&self, roots: &[PyObjectRef]) -> usize {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        let stack_slot = shadow_stack_cell();
        // SAFETY: this thread's cell, alive for the bracket; `incr_stack`
        // returns the slot it just claimed, and `publish` semantics claim
        // every index in the run `normalize_published_run` then reads.
        unsafe {
            let stack = &*stack_slot;
            let base = stack.len();
            for &root in roots {
                *stack.incr_stack() = root;
            }
            let _ = normalize_published_run(stack, base, roots.len());
            base
        }
    }

    /// One-word residual ABI for [`Self::publish`], the
    /// [`publish_roots_jit_abi`] twin.
    #[majit_macros::dont_look_inside_cannot_raise]
    pub extern "C" fn publish_jit_abi(
        &self,
        array: *const crate::object_array::GcTypedArray,
    ) -> i64 {
        let items = gcarray_ref_items(array);
        self.publish(&items) as i64
    }

    /// One-word residual ABI for [`Self::pin_roots`], the
    /// [`publish_roots_jit_abi`] twin.
    #[majit_macros::dont_look_inside_cannot_raise]
    pub extern "C" fn pin_roots_jit_abi(
        &self,
        array: *const crate::object_array::GcTypedArray,
    ) -> i64 {
        let items = gcarray_ref_items(array);
        self.pin_roots(&items) as i64
    }

    /// Scope-local [`normalize_roots`] on this thread's root-stack cell.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    pub fn normalize(&self, base: usize, len: usize) {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        let stack_slot = shadow_stack_cell();
        // SAFETY: `publish` claimed every index in this range, and
        // `stack_slot` is this thread's live root-stack cell.
        let _ = normalize_published_run(unsafe { &*stack_slot }, base, len);
    }

    /// [`normalize`](Self::normalize), reporting whether any slot in the run
    /// named a forwarding stub.
    ///
    /// A caller that has to rebuild a native view of the run from the slots
    /// only owes that rebuild when the run actually moved; `false` says the
    /// words it already holds are the live ones.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    pub fn normalize_moved(&self, base: usize, len: usize) -> bool {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        let stack_slot = shadow_stack_cell();
        // SAFETY: this thread's live cell; `publish` claimed the range.
        normalize_published_run(unsafe { &*stack_slot }, base, len)
    }

    /// Scope-local [`shadow_stack_set`] on this thread's root-stack cell — the
    /// write a slot whose contents change over the bracket takes on each update.
    #[inline]
    #[majit_macros::dont_look_inside_cannot_raise]
    pub fn set(&self, index: usize, root: PyObjectRef) {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        let stack_slot = shadow_stack_cell();
        // Publish the raw value before the query, for the reason [`pin_root`]
        // gives: the slot is what a walker reads, so it must name the value
        // before anything consults the collector about it. One cell
        // resolution covers the store and the normalize.
        // SAFETY: same cell; `slot` bounds-checks `index`.
        unsafe {
            let stack = &*stack_slot;
            *stack.slot(index) = root;
            normalize_published_slot(stack, index);
        }
    }
}

impl Drop for RootScope {
    #[inline]
    fn drop(&mut self) {
        root_scope_close(self);
    }
}

/// Truncate the shadow stack to the length captured at construction time,
/// mirroring `pop_roots`'s discard of the bracketed `livevars` slots.
///
/// This is the bracket's close spelled as a call taking the guard by
/// reference. A crate that sees `RootScope` only as an opaque cross-crate
/// stub cannot read `save_point` to close a bracket inline, but it can name
/// this function; the field read then happens behind it.
///
/// A residual for the same reason `pin_root` and `shadow_stack_cell_truncate`
/// are: the bracket's helpers are named by the runtime rather than looked
/// inside, and only a named helper resolves from a crate that carries no
/// declaration of it.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn root_scope_close(scope: &RootScope) {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    let cell = shadow_stack_cell();
    // `pop_roots` closes brackets innermost first.  A save point above the
    // live length means an enclosing bracket already closed and discarded
    // this one's pins: the guard, or a value owning it, outlived that
    // bracket, and every slot it reads back is gone.
    debug_assert!(
        scope.save_point <= shadow_stack_cell_len(cell),
        "a root scope closed after a bracket that enclosed it"
    );
    // `truncate` is a no-op if `save_point >= len()`, which is
    // the steady-state case for an empty bracket.
    shadow_stack_cell_truncate(cell, scope.save_point);
}

/// Open a `push_roots(hop)` bracket. Drop the returned guard to
/// execute the matching `pop_roots(hop, livevars)`. See the module
/// docstring for the multi-phase plan.
///
/// Residualised: `RootScope` is one word (`save_point`), which the residual
/// ABI can carry.  `rlib/jit.py` `@dont_look_inside`.
#[inline]
#[majit_macros::dont_look_inside]
pub fn push_roots() -> RootScope {
    RootScope::new()
}

/// A set of freshly allocated items held as GC roots while the rest of the
/// set is still being built.
///
/// A plain `Vec<PyObjectRef>` is not a root area, so an element sitting in one
/// while a later element allocates is unreachable to the collector: old-gen is
/// mark-sweep, and an element with no heap edge yet is sweepable rather than
/// merely immobile. `build_list_storage` states the resulting rule for its
/// callers -- every element must be pinned, and stay pinned across the block
/// allocation the constructor itself performs.
///
/// Slots are addressed as `base + i`, so only ONE set may be open at a time on
/// a thread: a second bracket's pins land above this one's base and take the
/// slots it expects, which would hand back the wrong elements rather than
/// unrooted ones. Build a nested set inside a block that closes before the
/// outer `push`, as the `_socket` and `pyexpat` conversions do. A debug build
/// asserts this.
///
/// [`take`](Self::take) borrows, so the bracket outlives the statement that
/// builds the container:
///
/// ```ignore
/// let mut items = RootedItems::new();
/// while ... {
///     items.push(allocating_call(...));
/// }
/// w_list_new(items.take())
/// ```
pub struct RootedItems {
    scope: RootScope,
    base: usize,
    len: usize,
}

impl RootedItems {
    /// Open a bracket for a set that is about to be built.
    #[inline]
    pub fn new() -> Self {
        let scope = push_roots();
        let base = scope.base();
        Self {
            scope,
            base,
            len: 0,
        }
    }

    /// Pin one more item. Pinning is what makes the earlier ones survive this
    /// item's allocation, so push each as it is produced rather than in a
    /// second pass over a `Vec`.
    #[inline]
    pub fn push(&mut self, item: PyObjectRef) {
        self.assert_owns_the_top();
        let _ = self.scope.pin_root(item);
        self.len += 1;
    }

    /// The slots this set reads back are `base..base + len`, which is only
    /// what it pinned while nothing else pins above it.
    #[inline]
    fn assert_owns_the_top(&self) {
        debug_assert_eq!(
            shadow_stack_len(),
            self.base + self.len,
            "another root bracket is open above this one and its pins take these slots"
        );
    }

    /// How many items have been pinned.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Live word at `index`.  Reloads the slot so a collect between two
    /// readers sees the forwarded pointer.  Unlike [`take`], this does not
    /// require the set to own the top of the stack.
    #[inline]
    pub fn get(&self, i: usize) -> PyObjectRef {
        assert!(
            i < self.len,
            "RootedItems::get index {i} out of range (len {})",
            self.len
        );
        self.scope.get(self.base + i)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read every pinned item back out of its slot. A relocation between the
    /// pin and this read rewrote the slot in place, so these are the live
    /// words. The bracket stays open -- the caller's constructor allocates.
    pub fn take(&self) -> Vec<PyObjectRef> {
        self.assert_owns_the_top();
        (0..self.len)
            .map(|i| self.scope.get(self.base + i))
            .collect()
    }
}

/// Process-global GCREF slot, the translator's static root for a cached
/// heap type. `OnceLock<usize>` stores an address the collector cannot
/// rewrite; this slot is registered with MiniMark on first init.
pub struct RootedOnceRef {
    slot: std::cell::UnsafeCell<usize>,
    init: std::sync::Mutex<()>,
    ready: std::sync::atomic::AtomicBool,
    registered: std::sync::atomic::AtomicBool,
}

// The slot is written once under `init`, then only by the collector.
unsafe impl Sync for RootedOnceRef {}
unsafe impl Send for RootedOnceRef {}

impl RootedOnceRef {
    pub const fn new() -> Self {
        Self {
            slot: std::cell::UnsafeCell::new(0),
            init: std::sync::Mutex::new(()),
            ready: std::sync::atomic::AtomicBool::new(false),
            registered: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn try_register(&self) -> bool {
        if self
            .registered
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            return true;
        }
        let ok = unsafe {
            crate::gc_hook::try_gc_add_root(self.slot.get() as *mut crate::gc_hook::GCREF)
        };
        if !ok {
            self.registered
                .store(false, std::sync::atomic::Ordering::Release);
        }
        ok
    }

    /// True when this slot may cache a pointer: either it is a registered
    /// MiniMark root, or no add-root hook exists so nothing can collect.
    fn can_publish(&self) -> bool {
        self.try_register() || !crate::gc_hook::add_root_hook_installed()
    }

    pub fn get_or_init(&self, init: impl FnOnce() -> PyObjectRef) -> PyObjectRef {
        if self.ready.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.try_register();
            return unsafe { *self.slot.get() as PyObjectRef };
        }
        let _guard = self.init.lock().unwrap_or_else(|e| e.into_inner());
        if self.ready.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.try_register();
            return unsafe { *self.slot.get() as PyObjectRef };
        }
        // Register the empty slot first. A hook that declines it leaves
        // the cell uninitialized so a later call retries; publishing
        // first would cache a pointer the collector cannot rewrite.
        if !self.can_publish() {
            return init();
        }
        let value = init();
        unsafe {
            *self.slot.get() = value as usize;
        }
        self.ready.store(true, std::sync::atomic::Ordering::Release);
        value
    }

    pub fn get(&self) -> Option<PyObjectRef> {
        if !self.ready.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        let _ = self.try_register();
        Some(unsafe { *self.slot.get() as PyObjectRef })
    }

    pub fn set(&self, value: PyObjectRef) {
        if self.ready.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.try_register();
            return;
        }
        let _guard = self.init.lock().unwrap_or_else(|e| e.into_inner());
        if self.ready.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.try_register();
            return;
        }
        if !self.can_publish() {
            return;
        }
        unsafe {
            *self.slot.get() = value as usize;
        }
        self.ready.store(true, std::sync::atomic::Ordering::Release);
    }
}

impl Default for RootedItems {
    fn default() -> Self {
        Self::new()
    }
}

/// Pin a [`PyObjectRef`] as a live GC root for the duration of the
/// surrounding [`RootScope`]. Mirrors RPython's per-livevar
/// `setarrayitem(rootstk, idx, gcref)` emitted by
/// `gct_fv_gc_malloc.push_roots`.
///
/// Callers should always pair this with a guard from [`push_roots`]
/// so the matching [`Drop`] truncates the entry. Pinning without a
/// guard is a leak from the GC's perspective once the backend GC consumes the stack.
///
/// Returns the word now held by the shadow-stack slot. A collection may have
/// forwarded `root` after the caller copied it, so callers that use the word
/// after pinning must use this value.
///
/// Pushes onto the thread-local `ROOT_STACK`, a runtime-mutable root the
/// tracer cannot type; the JIT residualises the call instead of tracing into
/// it (`@dont_look_inside`, `rlib/jit.py`), the `shadow_stack_len` twin.
#[must_use = "a pinned root may have been normalized; use the returned live word or bind it to `let _ =` for liveness-only pins"]
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn pin_root(root: PyObjectRef) -> PyObjectRef {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    // Publish the raw value first.  `try_gc_current_object_address` is itself
    // a GC operation and may wait behind another thread's collection; the
    // fresh allocation must already be visible to that collector before we
    // enter the query safepoint.  RPython's push_roots writes the livevar to
    // the shadow stack before calling the allocation/GC slow path for the
    // same reason.  One `with_shadow_stack` resolves the thread-local cell
    // once for both the push and the normalize, matching `RootScope::pin_root`.
    with_shadow_stack(|stack| {
        let index = stack.len();
        // SAFETY: `incr_stack` returns the slot it just claimed.
        unsafe { *stack.incr_stack() = root };
        normalize_published_slot(stack, index)
    })
}

/// Reload the current top shadow-stack entry after a call that may have moved
/// it. `root` names the value the top slot was published with; the runtime read
/// ignores that copy and answers the slot, which a collection forwards in
/// place.
///
/// The call stays a real residual, bound in `jit_trace_fnaddrs()`: aliasing it
/// to its argument would leave a trace carrying the pre-move word, and nothing
/// else forwards a value the trace holds.
#[inline]
#[must_use = "a top root may have moved; use the reloaded live word"]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn reload_top_root(root: PyObjectRef) -> PyObjectRef {
    let _ = root;
    majit_gc::shadow_stack::top_ref().0 as PyObjectRef
}

/// `extern "C"` residual entry for [`reload_top_root`], with the same
/// exception effect: the body only reads the shadow-stack top.
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn reload_top_root_jit_abi(root: PyObjectRef) -> PyObjectRef {
    reload_top_root(root)
}

/// Publish a complete translated livevar set before performing any
/// forwarding query.  RPython's `push_roots(hop)` writes every live GCREF to
/// the root stack as one pre-allocation phase; calling [`pin_root`] repeatedly
/// would query after the first write and let a foreign collection run before
/// later values were visible.
///
/// Returns the first shadow-stack index occupied by `roots`.
///
/// A livevar set that does not sit in one slice publishes each slice with
/// [`publish_roots`] and normalizes the whole range once with
/// [`normalize_roots`]. Calling this function per slice would reopen the
/// window it exists to close: the first call's queries are safepoints, and
/// the later slices are not yet on the stack when they run.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn pin_roots(roots: &[PyObjectRef]) -> usize {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    with_shadow_stack(|stack| {
        let base = stack.len();
        for &root in roots {
            // SAFETY: `incr_stack` returns the slot it just claimed.
            unsafe { *stack.incr_stack() = root };
        }
        let _ = normalize_published_run(stack, base, roots.len());
        base
    })
}

/// The write half of `push_roots(hop)`: store `roots` in fresh root-stack
/// slots and return the first index, querying the GC for nothing.
///
/// Split out for the callers whose livevar set spans several slices — they run
/// every `publish_roots` before the first [`normalize_roots`], so no value is
/// still invisible to a foreign collector once this mutator starts entering
/// safepoints.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn publish_roots(roots: &[PyObjectRef]) -> usize {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    with_shadow_stack(|stack| {
        let base = stack.len();
        for &root in roots {
            // SAFETY: `incr_stack` returns the slot it just claimed.
            unsafe { *stack.incr_stack() = root };
        }
        base
    })
}

/// One-word residual ABI for [`publish_roots`].
///
/// `&[PyObjectRef]` is `Ptr(GcArray(Ptr(PyObject)))`: one word, length at
/// offset 0 (`bh_newtuple_from_array`). The executor passes that word, not a
/// fat `(ptr, len)` pair.
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn publish_roots_jit_abi(array: *const crate::object_array::GcTypedArray) -> i64 {
    let items = gcarray_ref_items(array);
    publish_roots(&items) as i64
}

/// One-word residual ABI for [`pin_roots`], the [`publish_roots_jit_abi`]
/// twin.
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn pin_roots_jit_abi(array: *const crate::object_array::GcTypedArray) -> i64 {
    let items = gcarray_ref_items(array);
    pin_roots(&items) as i64
}

/// Flow-graph entry for a `Vec<PyObjectRef>` used as `&[PyObjectRef]`.
///
/// The vec is the address of its header. Slice consumers read one
/// length-prefixed object array. The residual call is
/// [`gcarray_from_pyobject_vec_jit_abi`].
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn gcarray_from_pyobject_vec(
    items: &Vec<PyObjectRef>,
) -> *mut crate::object_array::GcTypedArray {
    let _ = items;
    std::ptr::null_mut()
}

/// `ItemsBlock` and `GcTypedArray` share the length word at offset 0 and the
/// item base after it, so an items block is the array the slice readers use.
const _: () = {
    assert!(
        crate::object_array::ITEMS_BLOCK_LEN_OFFSET
            == crate::object_array::GC_TYPED_ARRAY_LEN_OFFSET
    );
    assert!(
        crate::object_array::ITEMS_BLOCK_ITEMS_OFFSET
            == crate::object_array::GC_TYPED_ARRAY_ITEMS_OFFSET
    );
};

/// Copy `items` into an exact-size type-9 items block.
///
/// `items` is the address of a live `Vec<PyObjectRef>`. The items are pinned
/// first, then [`crate::object_array::alloc_tuple_items_block_gc`] fills the
/// block from those slots (`cap == len`, including an empty vec). The length
/// word is that count. The block is not freed here: later slice reads use the
/// same word. With no collector hook the allocator falls back to `std::alloc`.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "the residual word is the address of a live Vec<PyObjectRef>"
)]
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn gcarray_from_pyobject_vec_jit_abi(
    items: *const u8,
) -> *mut crate::object_array::GcTypedArray {
    if items.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `items` is the address of a live `Vec<PyObjectRef>`. The vec
    // buffer is not a root; `pin_roots` copies every word onto the shadow
    // stack before the block allocation can collect.
    let vec = unsafe { &*(items as *const Vec<PyObjectRef>) };
    let _roots = push_roots();
    let save = pin_roots(vec);
    let block = unsafe { crate::object_array::alloc_tuple_items_block_gc(save, vec.len()) };
    block as *mut crate::object_array::GcTypedArray
}

/// Copy the items of a length-prefixed ref `GcTypedArray` word.
pub fn gcarray_ref_items(arr: *const crate::object_array::GcTypedArray) -> Vec<PyObjectRef> {
    let len = crate::object_array::gcarray_len(arr);
    let mut items = Vec::with_capacity(len);
    for index in 0..len {
        items.push(crate::object_array::getarrayitem_ref(arr, index));
    }
    items
}

/// Resolve forwarding across the `len` published slots starting at `base` —
/// the half of the pin that is itself a GC operation. See [`pin_root`] for why
/// a value copied into the bracket from outside it can already name a
/// forwarded nursery object.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn normalize_roots(base: usize, len: usize) {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    with_shadow_stack(|stack| {
        let _ = normalize_published_run(stack, base, len);
    });
}

/// Current length of the thread-local shadow stack. Used by
/// [`RootScope::new`] to capture the save-point and by tests to
/// observe pin/pop behaviour.
///
/// Reads the thread-local `ROOT_STACK`, a runtime-mutable root the
/// tracer cannot type; the JIT residualises the read instead of tracing
/// into it (`@dont_look_inside`, `rlib/jit.py`). The attribute is a
/// tracing-policy marker only — it leaves the host backend free to inline
/// this body, exactly as the RPython decorator leaves the C backend free.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_len() -> usize {
    with_shadow_stack(RootStack::len)
}

/// The thread's root-stack cell.  The JIT residualises the resolution instead
/// of tracing into it (`@dont_look_inside`, `rlib/jit.py`), the
/// `shadow_stack_len` twin.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_cell() -> *const RootStack {
    with_shadow_stack(|stack| stack as *const RootStack)
}

#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_cell_len(cell: *const RootStack) -> usize {
    // SAFETY: `cell` is this thread's root-stack cell, and `_not_send` keeps
    // the bracket on the thread that resolved it.
    unsafe { (*cell).len() }
}

#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_cell_truncate(cell: *const RootStack, len: usize) {
    // SAFETY: `cell` is this thread's root-stack cell, and `_not_send` keeps
    // the bracket on the thread that resolved it.
    unsafe { (*cell).truncate(len) };
}

/// Read a single shadow-stack slot by index, panicking if the index
/// is out of bounds.  RPython's `pop_roots` reads the root-stack slot
/// directly: a moving collection has already rewritten it in place through
/// [`walk_shadow_stack`] (or [`walk_shadow_stack_area`] for another mutator).
/// The initial [`pin_root`] still normalizes a copied pointer that may have
/// become stale before it was published.
///
/// This is the `pop_roots` half of the bracket — the read back of a pinned
/// livevar after a call that may have collected — so it sits on the ordinary
/// call path (`call_function_impl_result`, the inline-call and specialize
/// resume paths, `#[pyre_class]` method bodies), not just in tests.
///
/// Reads the thread-local `ROOT_STACK` the tracer cannot type; the JIT
/// residualises the read instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py`), the [`shadow_stack_len`] twin.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_get(index: usize) -> PyObjectRef {
    // A plain slot read: the slot is already a registered root, so whatever
    // moved it has rewritten it here.  `init_gc_subsystem` registers this
    // thread's stack as a mutator root area (`capture_shadow_stack_area`)
    // together with `register_mutator`, before the thread runs any user code,
    // and [`walk_shadow_stack`] hands the collector `&mut` slots — so a
    // relocation between the pin and this read updates the slot in place and
    // no forwarding stub can be observed here.
    //
    // Root publication APIs normalize, and they must: the value they receive
    // can be copied from outside the bracket before a collection and reach the
    // new slot as a forwarding stub.  A read allocates nothing and has no
    // reason to park.
    // SAFETY: `slot` bounds-checks `index` against the live length.
    with_shadow_stack(|stack| unsafe { *stack.slot(index) })
}

/// Copy a contiguous range of rooted values out of the shadow stack.
///
/// RPython's `pop_roots` reloads all livevars from the root stack as one
/// generated block after the potentially-collecting call.  Host-side callers
/// that need several adjacent roots should use this equivalent bulk shape
/// instead of repeatedly entering TLS through [`shadow_stack_get`].
///
/// Panics when `base..base + dst.len()` is outside the live root stack, just
/// like indexing the corresponding slots individually.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_copy_range(base: usize, dst: &mut [PyObjectRef]) {
    with_shadow_stack(|stack| {
        assert!(
            base + dst.len() <= stack.len(),
            "shadow-stack range out of bounds"
        );
        if dst.is_empty() {
            // An empty range starting one past the last live slot is in
            // bounds — `&stack[len..len]` was — but `slot(len)` is not, and
            // `copy_nonoverlapping` wants a non-null source even for a
            // zero-length copy.  Neither applies when there is nothing to
            // copy.
            return;
        }
        // SAFETY: the whole range was just bounds-checked, and `dst` is a
        // distinct caller-owned slice.
        unsafe { std::ptr::copy_nonoverlapping(stack.slot(base), dst.as_mut_ptr(), dst.len()) };
    });
}

/// [`shadow_stack_copy_range`] into a `Vec<PyObjectRef>`.
///
/// `&mut vec[..]` is that vec. The traced call passes the header address,
/// and the items are the vec's own buffer.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_copy_range_into_vec(base: usize, dst: &mut Vec<PyObjectRef>) {
    shadow_stack_copy_range(base, dst.as_mut_slice());
}

/// Copy `src[start..start + dst.len()]` into `dst`.
///
/// The erased form of [`shadow_stack_copy_range`] when the slots were an
/// incoming `&[PyObjectRef]` the root-bracket pass took out of the jitcode
/// (`RootBracketPlan`; `gctransform` / `shadowstack.py push_roots` never
/// emits a bracket into a jitcode). `W_CData.call` (`cdataobj.py descr_call`)
/// passes `args_w` through; the wrapper's `args[1:]` copy is that list slice.
///
/// `copy_from_slice` is a residual memmove and needs a concrete `src`
/// pointer. The recording walk of `__majit_wrap_cdata_call` holds the
/// incoming args array as a virtual `GcArray` (`box_value=None`), so that
/// residual aborts `ResidualCallArgUnbound`. An `@unroll_safe` item loop
/// is `rlist.py ll_arraycopy` / `rgc.py ll_arraycopy` for length 1 (`copy_item`):
/// `getarrayitem` reads the virtual source through heapcache and
/// `setarrayitem` writes the dest the vec-alloc residual already bound.
#[inline]
#[majit_macros::unroll_safe]
pub fn copy_object_slice_range_into_vec(
    src: &[PyObjectRef],
    start: usize,
    dst: &mut Vec<PyObjectRef>,
) {
    let n = dst.len();
    let mut i = 0usize;
    while i < n {
        dst[i] = src[start + i];
        i += 1;
    }
}

/// One-word residual ABI for [`copy_object_slice_range_into_vec`].
///
/// `src` is `Ptr(GcArray(Ptr(PyObject)))`. `dst` is the address of a live
/// `Vec<PyObjectRef>`.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "the residual words are a live GcTypedArray and a live Vec header"
)]
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn copy_object_slice_range_into_vec_jit_abi(
    src: *const crate::object_array::GcTypedArray,
    start: usize,
    dst: *mut u8,
) {
    if dst.is_null() {
        return;
    }
    let vec = unsafe { &mut *(dst as *mut Vec<PyObjectRef>) };
    let n = vec.len();
    if n == 0 {
        return;
    }
    let src_len = crate::object_array::gcarray_len(src);
    assert!(
        start <= src_len && n <= src_len - start,
        "object-slice range out of bounds"
    );
    for i in 0..n {
        vec[i] = crate::object_array::getarrayitem_ref(src, start + i);
    }
}

/// Overwrite a single shadow-stack slot by index, panicking if the index
/// is out of bounds. A slot whose contents change over a bracket's lifetime
/// is written here rather than re-pinned, mirroring `gc_save_root`'s
/// overwrite of an already allocated slot (`shadowcolor.py _gc_save_root`).
///
/// Writes the thread-local `ROOT_STACK` the tracer cannot type; the JIT
/// residualises the write instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py`), the [`shadow_stack_get`] twin.
#[inline]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn shadow_stack_set(index: usize, root: PyObjectRef) {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    // Publish the raw value first, for the reason [`pin_root`] gives: the
    // `try_gc_current_object_address` query is itself a GC operation and may
    // wait behind another thread's collection, so the value must already be
    // visible to that collector before we enter the query safepoint.
    // SAFETY: `slot` bounds-checks `index` against the live length.
    // One cell resolution covers the store and the normalize, matching
    // [`RootScope::set`].
    with_shadow_stack(|stack| {
        unsafe { *stack.slot(index) = root };
        normalize_published_slot(stack, index);
    });
}

/// Visit every pinned root in the shadow stack with mutable access.
/// `pyre-jit/src/eval.rs` registers a thin adapter through
/// `majit_gc::shadow_stack::register_extra_root_walker` that forwards
/// `&mut PyObjectRef` slots into the GC's `&mut GcRef` visitor; the
/// two are layout-compatible because `GcRef` is
/// `#[repr(transparent)]` over `usize` and `PyObjectRef =
/// *mut PyObject` is also pointer-sized.
///
/// The visitor receives mutable references so a moving collector can rewrite
/// slot contents post-relocation. Each slot is re-derived from the stack after
/// the previous visitor call, so a re-entrant push cannot leave the walk using
/// a pointer into an old allocation. Debug builds additionally reject shadow-
/// stack mutation by an accidentally allocating walker; release builds match
/// upstream's raw root-stack access without that diagnostic.
#[inline]
pub fn walk_shadow_stack(mut visitor: impl FnMut(&mut PyObjectRef)) {
    ROOT_STACK.with(|stack| {
        let cell = stack as *const RootStack;
        #[cfg(debug_assertions)]
        SHADOW_STACK_WALK_IN_PROGRESS.with(|in_progress| {
            let _walk = ShadowStackWalk::new(in_progress);
            // SAFETY: `cell` is this thread's live shadow-stack cell.
            unsafe { walk_shadow_stack_cell(cell, &mut visitor) };
        });
        #[cfg(not(debug_assertions))]
        // SAFETY: `cell` is this thread's live shadow-stack cell.
        unsafe {
            walk_shadow_stack_cell(cell, &mut visitor);
        }
    });
}

/// Capture the current thread's pinned-root stack for a foreign STW walk.
pub fn capture_shadow_stack_area() -> *const () {
    // The captured address is the `RootStack` cell itself, never its buffer:
    // `grow` replaces the buffer, so a foreign walker must re-read base/top —
    // the same rule `majit-gc`'s jitframe root stack states for the
    // `root_stack_top` address it hands to compiled code.
    ROOT_STACK.with(|s| s as *const _ as *const ())
}

/// Visit a captured thread's pinned-root stack without consulting caller TLS.
/// Like [`walk_shadow_stack`], this re-derives each slot after the previous
/// visitor call so a push cannot strand the walk in a reallocated buffer.
/// Debug builds reject shadow-stack mutation for the duration of the walk.
///
/// # Safety
/// `data` must come from [`capture_shadow_stack_area`], and the owning thread
/// must be quiesced for the duration of the walk.
pub unsafe fn walk_shadow_stack_area(data: *const (), mut visitor: impl FnMut(&mut PyObjectRef)) {
    let cell = data as *const RootStack;
    #[cfg(debug_assertions)]
    SHADOW_STACK_WALK_IN_PROGRESS.with(|in_progress| {
        // For a self-collection this protects the stack being walked. A
        // foreign walk marks the collector thread instead, which is harmless:
        // the owning mutator is quiesced and performs no stack accesses.
        let _walk = ShadowStackWalk::new(in_progress);
        // SAFETY: the caller guarantees that `cell` is a captured shadow stack
        // whose owning thread is quiesced for this walk.
        unsafe { walk_shadow_stack_cell(cell, &mut visitor) };
    });
    #[cfg(not(debug_assertions))]
    // SAFETY: the caller guarantees that `cell` is a captured shadow stack
    // whose owning thread is quiesced for this walk.
    unsafe {
        walk_shadow_stack_cell(cell, &mut visitor);
    }
}

/// Walk one captured root stack, re-reading `base` for every slot.
///
/// # Safety
/// `cell` must remain a valid root-stack cell for the duration of the walk,
/// and no thread other than a re-entrant visitor may mutate it.
unsafe fn walk_shadow_stack_cell(
    cell: *const RootStack,
    visitor: &mut impl FnMut(&mut PyObjectRef),
) {
    // The interval is fixed at entry: `walk_stack_root` (`shadowstack.py`)
    // receives `start` and `addr` as arguments and runs `while addr != start`,
    // so a root pushed mid-walk is not part of that walk. Re-reading the length
    // each iteration instead would extend the walk over roots pinned by the
    // visitor — and pinning during a walk is precisely what the debug guard
    // above declares illegal, so there is nothing to serve by diverging here.
    // SAFETY: the caller guarantees `cell` names a live root stack.
    let stack = unsafe { &*cell };
    let end = stack.len();
    let mut index = 0;
    while index < end {
        // Re-read `base` every iteration rather than caching it: a re-entrant
        // push that hits the limit calls `grow`, which moves the buffer and
        // would strand a cached pointer in the freed one. Only pushes can
        // occur, so `end` stays within bounds.
        // SAFETY: `index < end <= len()`, so the slot is live.
        let mut root = unsafe { *stack.base.get().add(index) };
        visitor(&mut root);
        // Store back through a re-read `base` rather than handing the visitor
        // a `&mut` into the buffer: a re-entrant push that grows the stack
        // frees the buffer the slot was read from, and a visitor that writes
        // its reference after allocating would then write into the freed one.
        // `grow` copies the live prefix, so `index` still names this slot.
        // SAFETY: `index < end <= len()`, so the slot is live.
        unsafe { *stack.base.get().add(index) = root };
        index += 1;
    }
}

// ── Prebuilt-root write tracking ────────────────────────────────────
//
// incminimark.py `GCFLAG_TRACK_YOUNG_PTRS` / 339-344
// `old_objects_pointing_to_young` / 355 `prebuilt_root_objects`: an old or
// prebuilt object is NOT scanned during a minor collection unless the write
// barrier recorded a store into it since the previous minor collection; a
// major collection always traces the prebuilt roots.
//
// pyre's Box-immortal interpreter structures (module dicts / cells, heap-type
// namespace dicts, method caches, function fields) bypass the compiled-code
// write barrier — their stores go through Rust runtime helpers — so the
// band-aid root walks in `pyre-interpreter::eval::walk_pyframe_roots` rescan
// them at EVERY minor collection.  This flag is the write-barrier bit for
// that whole prebuilt family, one global bit instead of a per-object flag
// (coarser than upstream: any single store rescans everything on the next
// minor collection — still sound, just less selective).  Every mutation
// helper that can store a (possibly nursery-young) `PyObjectRef` into one of
// those walked structures must call [`mark_prebuilt_roots_dirty`]; the minor
// walk clears the bit after it runs (the walk itself promotes every young
// value it reaches, so a clean bit again means "no young pointers inside").
use std::sync::atomic::{AtomicBool, Ordering};

static PREBUILT_ROOTS_DIRTY: AtomicBool = AtomicBool::new(true);

/// Write-barrier hook for the prebuilt/Box-immortal root family
/// (incminimark `remember_young_pointer` analog).  Call from every helper
/// that stores a `PyObjectRef` into a structure reached only by the
/// `walk_pyframe_roots` band-aid walks.
///
/// Sets the static `PREBUILT_ROOTS_DIRTY` bit the tracer cannot model; the
/// JIT residualises the call instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py`), the `pin_root` twin.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn mark_prebuilt_roots_dirty() {
    PREBUILT_ROOTS_DIRTY.store(true, Ordering::Relaxed);
}

/// Whether any prebuilt-family store happened since the last completed
/// minor-collection walk (incminimark: "is this object in
/// `old_objects_pointing_to_young`?").
#[inline]
pub fn prebuilt_roots_dirty() -> bool {
    PREBUILT_ROOTS_DIRTY.load(Ordering::Relaxed)
}

/// Clear the write-tracking bit after a minor-collection walk promoted every
/// young value reachable from the prebuilt family (incminimark: the minor
/// collection empties `old_objects_pointing_to_young` and re-sets
/// GCFLAG_TRACK_YOUNG_PTRS).  Must only be called by the walker, after the
/// walk completed, during a stop-the-world collection.
#[inline]
pub fn clear_prebuilt_roots_dirty() {
    PREBUILT_ROOTS_DIRTY.store(false, Ordering::Relaxed);
}

/// `shadowstack.py push_roots` / `pop_roots` as a bracket around one
/// collecting operation.
///
/// Upstream never writes this pair by hand: `framework.py` brackets every
/// operation that can reach the collector, `shadowstack.py` emits the
/// `gc_push_roots` / `gc_pop_roots` pair, and `expand_pop_roots` turns the pop
/// into one `gc_restore_root` per variable -- whose whole body is
/// `setvar(v_value, newvalue)`, writing the forwarded pointer back into the
/// local the graph will read next. Application code therefore names its
/// receiver and its argument as plain locals and never sees a root stack.
///
/// The flowspace translator now inserts and colours this bracket from graph
/// liveness, just as PyPy does. Native interpreter code is also compiled
/// directly by rustc, so source paths on that execution side still use this
/// macro. It publishes `$local`s as one livevar set, assigns each forwarded
/// word back into its own local (`ShadowStackFrameworkGCTransformer.pop_roots`)
/// so `$body` does not see the pre-pin words — `pin_roots` is itself a
/// safepoint — and then assigns again after `$body`. What it buys over
/// hand-written slot arithmetic is `gc_restore_root`'s guarantee -- after the
/// bracket the local *is* the live word, so there is no stale binding left in
/// scope for a later read to find, and no slot index to get wrong.
///
/// Every `$local` must be a `mut` binding, because the restore assigns to it.
///
/// ```ignore
/// let mut obj = ...;
/// let mut key = ...;
/// let found = with_roots!(obj, key => lookup_that_can_collect(obj, key))?;
/// // `obj` and `key` are the forwarded words here, not the pre-call ones.
/// ```
///
/// The livevar set is published with [`RootScope::pin_roots`] rather than a
/// `pin_root` per local: that function's own docs give the reason -- the
/// first `pin_root`'s forwarding query is a safepoint, so a per-local loop
/// lets a foreign collection run while the later values are still invisible.
///
/// Keep `?` outside the macro.  A guard alive across an early return puts the
/// guard's close between the call and its `Result` switch, which the
/// `Result`-to-exception rewrite does not see through.
///
/// An early exit out of `$body` (a `?`, a `return`) skips the restore after
/// `$body` and that is correct: [`push_roots`]'s scope guard pops the slots
/// on the way out, and the locals it would have restored are going out of
/// scope with it. The restore after `pin_roots` has already run.
///
/// A macro rather than a helper taking `impl FnOnce`: a closure has no lifted
/// counterpart, so every graph that reached one would stop there. Bracketing a
/// collecting call is exactly what the hot paths do, and they are the graphs
/// the translator must be able to walk through.
#[macro_export]
macro_rules! with_roots {
    ($($local:ident),+ $(,)? => $body:expr) => {{
        let _scope = $crate::gc_roots::push_roots();
        let __roots_base = _scope.pin_roots(&[$($local),+]);
        #[allow(unused_assignments)]
        {
            $crate::__with_roots_restore!(_scope, __roots_base; $($local),+);
        }
        let __roots_result = $body;
        #[allow(unused_assignments)]
        {
            $crate::__with_roots_restore!(_scope, __roots_base; $($local),+);
        }
        __roots_result
    }};
}

/// The restore half of [`with_roots!`]: `$local = scope.get(base + k)` for
/// the `k`-th local, each index spelled as a literal offset from the scope's
/// base.  That is the read-back shape the jitcode lowering answers with the
/// pinned value (`RootBracketPlan`), so a bracket spelled through the macro
/// leaves the jitcode the same way a hand-written one does.
#[doc(hidden)]
#[macro_export]
macro_rules! __with_roots_restore {
    ($s:ident, $b:ident; $l0:ident) => {
        $l0 = $s.get($b);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident, $l4:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
        $l4 = $s.get($b + 4);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident, $l4:ident, $l5:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
        $l4 = $s.get($b + 4);
        $l5 = $s.get($b + 5);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident, $l4:ident, $l5:ident,
     $l6:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
        $l4 = $s.get($b + 4);
        $l5 = $s.get($b + 5);
        $l6 = $s.get($b + 6);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident, $l4:ident, $l5:ident,
     $l6:ident, $l7:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
        $l4 = $s.get($b + 4);
        $l5 = $s.get($b + 5);
        $l6 = $s.get($b + 6);
        $l7 = $s.get($b + 7);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident, $l4:ident, $l5:ident,
     $l6:ident, $l7:ident, $l8:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
        $l4 = $s.get($b + 4);
        $l5 = $s.get($b + 5);
        $l6 = $s.get($b + 6);
        $l7 = $s.get($b + 7);
        $l8 = $s.get($b + 8);
    };
    ($s:ident, $b:ident; $l0:ident, $l1:ident, $l2:ident, $l3:ident, $l4:ident, $l5:ident,
     $l6:ident, $l7:ident, $l8:ident, $l9:ident) => {
        $l0 = $s.get($b);
        $l1 = $s.get($b + 1);
        $l2 = $s.get($b + 2);
        $l3 = $s.get($b + 3);
        $l4 = $s.get($b + 4);
        $l5 = $s.get($b + 5);
        $l6 = $s.get($b + 6);
        $l7 = $s.get($b + 7);
        $l8 = $s.get($b + 8);
        $l9 = $s.get($b + 9);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(addr: usize) -> PyObjectRef {
        addr as PyObjectRef
    }

    #[test]
    fn gcarray_from_pyobject_vec_copies_the_vec() {
        let items = vec![dummy(0x10), dummy(0x20), dummy(0x30)];
        let array =
            gcarray_from_pyobject_vec_jit_abi(&items as *const Vec<PyObjectRef> as *const u8);
        assert_eq!(crate::object_array::gcarray_len(array), 3);
        assert_eq!(crate::object_array::getarrayitem_ref(array, 0), dummy(0x10));
        assert_eq!(crate::object_array::getarrayitem_ref(array, 1), dummy(0x20));
        assert_eq!(crate::object_array::getarrayitem_ref(array, 2), dummy(0x30));
        assert!(gcarray_from_pyobject_vec_jit_abi(std::ptr::null()).is_null());
        let empty: Vec<PyObjectRef> = Vec::new();
        let empty_array =
            gcarray_from_pyobject_vec_jit_abi(&empty as *const Vec<PyObjectRef> as *const u8);
        assert!(!empty_array.is_null());
        assert_eq!(crate::object_array::gcarray_len(empty_array), 0);
    }

    /// The scope guard is returned by value and dropped at end of
    /// the binding. This is the shape callers will rely on across
    /// every phase.
    #[test]
    fn push_roots_returns_a_drop_guard() {
        let _roots = push_roots();
    }

    /// `RootScope` carries only the saved top; the root-stack cell is
    /// re-resolved. Trap unintended growth beyond that one word.
    #[test]
    fn root_scope_payload_is_one_word() {
        assert_eq!(
            std::mem::size_of::<RootScope>(),
            std::mem::size_of::<usize>()
        );
    }

    /// The naive form costs one frame per bracket: each `with_roots!` pushes
    /// its own livevar set and pops it again, so the depth inside the second
    /// bracket is the same as inside the first and neither outlives it.
    #[test]
    fn hand_written_brackets_each_open_their_own_frame() {
        fn two_brackets(mut a: PyObjectRef, mut b: PyObjectRef) -> (usize, usize) {
            let first = crate::with_roots!(a, b => shadow_stack_len());
            let between = shadow_stack_len();
            let second = crate::with_roots!(a, b => shadow_stack_len());
            let _ = (a, b, second);
            (first, between)
        }
        let before = shadow_stack_len();
        let (first, between) = two_brackets(dummy(0x10), dummy(0x20));
        assert_eq!(first, before + 2, "the bracket publishes both livevars");
        assert_eq!(between, before, "and pops them before the next one opens");
        assert_eq!(shadow_stack_len(), before);
    }

    #[test]
    fn scope_local_pin_and_get_share_the_cached_stack_cell() {
        let before = shadow_stack_len();
        {
            let roots = push_roots();
            assert_eq!(roots.base(), before);
            let _ = roots.pin_root(dummy(0x1234));
            let _ = roots.pin_root(dummy(0x5678));
            assert_eq!(roots.get(before) as usize, 0x1234);
            assert_eq!(roots.get(before + 1) as usize, 0x5678);
        }
        assert_eq!(shadow_stack_len(), before);
    }

    /// `pin_root` appends, the matching `Drop` truncates back to the
    /// pre-bracket length. Mirrors the per-call invariant of
    /// `push_roots(livevars) ... pop_roots(livevars)` in
    /// `framework.py:803-853`.
    #[test]
    fn pin_root_pops_when_scope_drops() {
        let before = shadow_stack_len();
        {
            let _roots = push_roots();
            let _ = pin_root(dummy(0xDEAD_BEEF));
            let _ = pin_root(dummy(0xCAFE_F00D));
            assert_eq!(shadow_stack_len(), before + 2);
            assert_eq!(shadow_stack_get(before) as usize, 0xDEAD_BEEF);
            assert_eq!(shadow_stack_get(before + 1) as usize, 0xCAFE_F00D);
        }
        assert_eq!(shadow_stack_len(), before);
    }

    /// Nested brackets pop in LIFO order — RPython allows nested
    /// `push_roots` for cases like Container::new(member_alloc()).
    /// Each `Drop` rewinds to its own save-point regardless of inner
    /// pins.
    #[test]
    fn nested_root_scopes_pop_in_lifo_order() {
        let before = shadow_stack_len();
        let outer = push_roots();
        let _ = pin_root(dummy(1));
        let mid_outer_len = shadow_stack_len();
        assert_eq!(mid_outer_len, before + 1);
        {
            let _inner = push_roots();
            let _ = pin_root(dummy(2));
            let _ = pin_root(dummy(3));
            assert_eq!(shadow_stack_len(), mid_outer_len + 2);
        }
        // _inner dropped — back to the outer's view.
        assert_eq!(shadow_stack_len(), mid_outer_len);
        drop(outer);
        assert_eq!(shadow_stack_len(), before);
    }

    /// Empty bracket is a no-op against the stack length, both at
    /// `push` and at `Drop`. This protects callers that wrap an
    /// allocation that does not need to pin any of its inputs.
    #[test]
    fn empty_bracket_does_not_touch_the_stack() {
        let before = shadow_stack_len();
        {
            let _roots = push_roots();
            assert_eq!(shadow_stack_len(), before);
        }
        assert_eq!(shadow_stack_len(), before);
    }

    /// An empty range is in bounds anywhere up to and including the live
    /// length — `&stack[len..len]` was, and `pop_roots` is emitted even for
    /// zero livevars (`shadowstack.py`). The top-of-stack case is the
    /// one a per-slot bounds check gets wrong.
    #[test]
    fn copy_range_accepts_an_empty_range_at_the_top_of_the_stack() {
        let _roots = push_roots();
        let _ = pin_root(dummy(0x1));
        shadow_stack_copy_range(shadow_stack_len(), &mut []);
    }

    /// Past the live length it is out of bounds even when empty, which is
    /// also what indexing did.
    #[test]
    #[should_panic(expected = "shadow-stack range out of bounds")]
    fn copy_range_rejects_an_empty_range_past_the_top_of_the_stack() {
        let _roots = push_roots();
        let _ = pin_root(dummy(0x1));
        shadow_stack_copy_range(shadow_stack_len() + 1, &mut []);
    }

    #[test]
    fn copy_range_into_vec_reloads_the_vec() {
        let _roots = push_roots();
        let base = shadow_stack_len();
        let _ = pin_root(dummy(0x11));
        let _ = pin_root(dummy(0x22));
        let mut items = vec![dummy(0), dummy(0)];
        shadow_stack_copy_range_into_vec(base, &mut items);
        assert_eq!(items, vec![dummy(0x11), dummy(0x22)]);
        shadow_stack_copy_range_into_vec(shadow_stack_len(), &mut Vec::new());
    }

    #[test]
    fn copy_object_slice_range_into_vec_copies_from_a_gcarray() {
        let items = vec![dummy(0x10), dummy(0x20), dummy(0x30)];
        let array =
            gcarray_from_pyobject_vec_jit_abi(&items as *const Vec<PyObjectRef> as *const u8);
        let mut dst = vec![dummy(0), dummy(0)];
        copy_object_slice_range_into_vec_jit_abi(
            array,
            1,
            &mut dst as *mut Vec<PyObjectRef> as *mut u8,
        );
        assert_eq!(dst, vec![dummy(0x20), dummy(0x30)]);
        copy_object_slice_range_into_vec(&items, 0, &mut dst);
        assert_eq!(dst, vec![dummy(0x10), dummy(0x20)]);
    }

    /// `walk_shadow_stack` exposes every pinned slot with mutable
    /// access, in insertion order. Mirrors what the GC walker sees
    /// during collection — a moving collector that updates a slot
    /// must observe the write through the next `shadow_stack_get`.
    #[test]
    fn walk_shadow_stack_visits_each_slot_mutably() {
        let _roots = push_roots();
        let _ = pin_root(dummy(0x1));
        let _ = pin_root(dummy(0x2));
        let _ = pin_root(dummy(0x3));

        // Read pass: collect the addresses we see.
        let mut seen = Vec::new();
        walk_shadow_stack(|slot| seen.push(*slot as usize));
        // Trim to the entries we just pinned to avoid false matches
        // against entries pinned by a concurrent test (single-threaded
        // here, but the contract is per-thread).
        let tail = &seen[seen.len() - 3..];
        assert_eq!(tail, &[0x1, 0x2, 0x3]);

        // Write pass: a moving collector might rewrite slots — confirm
        // the writes stick.
        walk_shadow_stack(|slot| {
            if (*slot as usize) == 0x2 {
                *slot = dummy(0xAA);
            }
        });
        assert_eq!(shadow_stack_get(shadow_stack_len() - 2) as usize, 0xAA);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "the shadow stack was mutated while a root walk was in progress")]
    fn walk_shadow_stack_area_rejects_reentrant_push_in_debug_builds() {
        let _roots = push_roots();
        let _ = pin_root(dummy(0x1));
        let area = capture_shadow_stack_area();
        // SAFETY: `area` is this thread's shadow stack, and this synchronous
        // self-walk keeps it live for the duration of the call.
        unsafe {
            walk_shadow_stack_area(area, |_| {
                let _ = pin_root(dummy(0x2));
            })
        };
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn walk_shadow_stack_survives_push_that_grows_the_buffer() {
        let _roots = push_roots();
        let _ = pin_root(dummy(0x11));
        let _ = pin_root(dummy(0x22));

        let (capacity_before, additional) =
            with_shadow_stack(|stack| (stack.capacity(), stack.capacity() - stack.len() + 1));
        let mut pushed = false;
        let mut seen = Vec::new();
        walk_shadow_stack(|slot| {
            seen.push(*slot as usize);
            if !pushed {
                pushed = true;
                for offset in 0..additional {
                    let _ = pin_root(dummy(0x1000 + offset));
                }
            }
        });

        assert!(pushed);
        // The visitor's pins really did hit the limit and move the buffer.
        assert!(with_shadow_stack(RootStack::capacity) > capacity_before);
        // `0x22` is the discriminator: it is read on the iteration AFTER the
        // grow, so getting it right means the walk followed the buffer
        // to its new address instead of reading the freed one. The walk stops
        // at the two roots that were live at entry — the interval is fixed
        // there, as in `walk_stack_root` (`shadowstack.py`), so the
        // roots the visitor pinned are not part of this walk.
        assert_eq!(seen, vec![0x11, 0x22]);
        assert_eq!(shadow_stack_get(0) as usize, 0x11);
        assert_eq!(shadow_stack_get(1) as usize, 0x22);
    }

    /// The read twin above only proves the walk *reads* from the relocated
    /// buffer. A moving collector also *writes* the forwarded address back,
    /// and that write happens after the visitor has had its chance to grow
    /// the stack — so the store has to resolve the buffer address again
    /// rather than reuse the one the slot was read from.
    #[cfg(not(debug_assertions))]
    #[test]
    fn walk_shadow_stack_write_back_follows_a_grow_inside_the_visitor() {
        let _roots = push_roots();
        let _ = pin_root(dummy(0x11));
        let _ = pin_root(dummy(0x22));

        let additional = with_shadow_stack(|stack| stack.capacity() - stack.len() + 1);
        let mut pushed = false;
        walk_shadow_stack(|slot| {
            if !pushed {
                pushed = true;
                for offset in 0..additional {
                    let _ = pin_root(dummy(0x1000 + offset));
                }
            }
            // Forward the root the way a moving collector would, *after* the
            // buffer has already moved under this slot.
            *slot = dummy(*slot as usize + 0xA0_0000);
        });

        assert!(pushed);
        // Slot 0 is the discriminator: it was read from the old buffer, and
        // the grow happened before the visitor forwarded it. The write is
        // only observable here if it went to the buffer the stack owns now.
        // Slot 1 was read after the grow, so it lands correctly either way.
        assert_eq!(shadow_stack_get(0) as usize, 0xA0_0011);
        assert_eq!(shadow_stack_get(1) as usize, 0xA0_0022);
    }
}
