//! The deadframe a jitframe-backed backend returns from a compiled run.
//!
//! `llmodel.py return ll_frame` — the deadframe IS the JitFrame. Values
//! stay in `jf_frame[]` and are never copied out: `get_int_value(deadframe,
//! index)` (`llmodel.py`) casts the opaque deadframe back to a
//! JITFRAMEPTR and reads `jf_frame[index]` in place, and `get_latest_descr`
//! (`llmodel.py`) does the same for `jf_descr`.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use majit_gc::shadow_stack::ShadowRoot;
use majit_ir::{DescrRef, GcRef};

use crate::ExitRecoveryLayout;
use crate::jitframe::{FIRST_ITEM_OFFSET, JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS, JitFrame};

/// The host-heap frame a deadframe owns, and its release.
///
/// A descr with no `JITFRAME` type id builds its frames on the host heap
/// (`jitframe::malloc_jitframe`), one per compiled entry. The frame is not
/// read through this handle — the frame pointer the compiled code was handed
/// points into it — so all this type does is decide the lifetime, which is
/// why it is `_heap_owner` on the deadframe and never read there either.
///
/// Where the block goes on release is the free list's decision
/// ([`jitframe_pool_enabled`]): `jitframe_allocate` builds the frame out of
/// the GC nursery, a bump allocator with no per-frame release at all, so a
/// steady entry rate that pays the allocator nothing is the closer shape and
/// the default. Measured, on a compiled cel entry: pooled is 2.62 ns/entry
/// faster, negative in 24 of 24 shape-runs, and it makes the steady compiled
/// tier allocation-free (`allocs_per_eval` 1.000 -> 0.000). The owned arm —
/// one allocation and one free per entry — is the probe.
pub struct FrameHeapOwner {
    frame: *mut JitFrame,
}

impl FrameHeapOwner {
    /// Take over a host-heap frame from `jitframe::malloc_jitframe`.
    pub fn of(frame: *mut JitFrame) -> Self {
        FrameHeapOwner { frame }
    }
}

impl Drop for FrameHeapOwner {
    /// Release the frame, which for the pooled arm means parking its block.
    ///
    /// This runs when the deadframe holding it drops, and the deadframe is the
    /// last thing that reads the frame: the compiled run finished before the
    /// deadframe was built, and every accessor on it goes through
    /// `jf_gcref()`. So the interior pointer compiled code was handed is dead
    /// by the time the block is offered to the next entry. The pool hands a
    /// block out by REMOVING it from the free list, so two live frames can
    /// never be looking at one.
    fn drop(&mut self) {
        unsafe { crate::jitframe::free_host_jitframe(self.frame) };
    }
}

/// Frame blocks a thread keeps rather than frees.
///
/// Small because the count that matters is the number of frames live at once,
/// not the entry rate: entries are nested only by `execute_bridge` recursion
/// and the CALL_ASSEMBLER hop, so the steady state is one or two. Anything past
/// this is released to the allocator, which bounds a pathological trace's
/// footprint without costing the ordinary one anything.
const FRAME_POOL_CAPACITY: usize = 8;

/// A block parked on the free list: the base `jitframe::alloc_off_gc_jitframe`
/// got from the allocator, and the size its slot records.
#[derive(Clone, Copy)]
struct ParkedBlock {
    base: *mut u8,
    total: usize,
}

impl ParkedBlock {
    const EMPTY: ParkedBlock = ParkedBlock {
        base: std::ptr::null_mut(),
        total: 0,
    };
}

/// Frees a thread's parked blocks when the thread exits.
///
/// `FRAME_POOL` has no destructor so that its access path stays the direct
/// one; this key has one, and is touched exactly once per thread, on the
/// first park, which is what registers it. Thread-local destructors run in
/// no fixed order, but a key without one stays readable throughout, so the
/// pool is still there when this runs. A block parked after it ran — a
/// deadframe dropped during thread teardown — is the one case that leaks.
struct PoolReaper;

impl Drop for PoolReaper {
    fn drop(&mut self) {
        let _ = FRAME_SLOT.try_with(|slot| {
            let parked = slot.get();
            if !parked.base.is_null() {
                unsafe { crate::jitframe::dealloc_off_gc_block(parked.base, parked.total) };
                slot.set(SlotState::EMPTY);
            }
        });
        let _ = FRAME_POOL.try_with(|pool| {
            let mut pool = pool.borrow_mut();
            for parked in &pool.free[..pool.free_len] {
                unsafe { crate::jitframe::dealloc_off_gc_block(parked.base, parked.total) };
            }
            pool.free_len = 0;
        });
    }
}

thread_local! {
    static POOL_REAPER: PoolReaper = const { PoolReaper };
}

/// `Copy`, and so without a destructor: that is what lets `FRAME_POOL` be a
/// `const`-initialised thread-local with no lazy state and no registered
/// destructor, which is the fast access path. `POOL_REAPER` is the destructor,
/// kept on a key the hot path never touches.
#[derive(Clone, Copy)]
struct FramePool {
    free: [ParkedBlock; FRAME_POOL_CAPACITY],
    free_len: usize,
    /// Whether `POOL_REAPER` has been registered on this thread.
    reaper_armed: bool,
    /// Blocks the owned arm asked the allocator for.
    owned: u64,
    /// Blocks the pooled arm handed out.
    taken: u64,
    /// …of which the free list had nothing usable for, so the allocator was
    /// asked after all. `taken - misses` is what pooling actually saved.
    misses: u64,
}

/// One block kept in a `Cell`, ahead of [`FRAME_POOL`].
///
/// The steady portal holds a single frame size. Parking it here is one TLS
/// load and a pointer store. The `RefCell` list remains for a second live
/// frame (a nested entry) and for a size the slot cannot hold.
#[derive(Clone, Copy)]
struct SlotState {
    base: *mut u8,
    total: usize,
    reaper_armed: bool,
}

impl SlotState {
    const EMPTY: SlotState = SlotState {
        base: std::ptr::null_mut(),
        total: 0,
        reaper_armed: false,
    };
}

thread_local! {
    static FRAME_SLOT: Cell<SlotState> = const { Cell::new(SlotState::EMPTY) };
}

fn arm_frame_slot_reaper(slot: &Cell<SlotState>) {
    let mut state = slot.get();
    if state.reaper_armed {
        return;
    }
    state.reaper_armed = true;
    slot.set(state);
    let _ = POOL_REAPER.try_with(|_| ());
}

thread_local! {
    /// Per-thread, and per-thread is the only scope that pays for itself here.
    ///
    /// The effect being bought is 2.62 ns per entry. A pool shared between
    /// threads costs an uncontended atomic read-modify-write pair to take and to
    /// return a block, which is the same order as the saving, so a process-wide
    /// free list would hand back what it was built to remove. Sharding by thread
    /// is what leaves the take and the return as plain pops and pushes.
    ///
    /// A field would be the alternative to a `thread_local!`, and there is no
    /// object to make it a field of: `run_compiled_code` and
    /// `execute_token` are free functions on the backend, and the backend
    /// value itself is not on the entry path.
    static FRAME_POOL: RefCell<FramePool> = const {
        RefCell::new(FramePool {
            free: [ParkedBlock::EMPTY; FRAME_POOL_CAPACITY],
            free_len: 0,
            reaper_armed: false,
            owned: 0,
            taken: 0,
            misses: 0,
        })
    };
}

/// A parked block of at least `total` bytes, for `alloc_off_gc_jitframe`.
///
/// `None` when the arm is owned (tallied as such), the list is empty, or its
/// top block is too small — that one is released, so the list converges on
/// the tallest frame the thread has run and stops churning.
pub(crate) fn take_pooled_block(total: usize) -> Option<*mut u8> {
    if !jitframe_pool_enabled() {
        count_owned_frame_block();
        return None;
    }
    if let Some(base) = FRAME_SLOT
        .try_with(|slot| {
            let state = slot.get();
            if state.base.is_null() || state.total < total {
                return None;
            }
            slot.set(SlotState {
                base: std::ptr::null_mut(),
                total: 0,
                reaper_armed: state.reaper_armed,
            });
            Some(state.base)
        })
        .ok()
        .flatten()
    {
        return Some(base);
    }
    FRAME_POOL
        .try_with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.taken += 1;
            if pool.free_len == 0 {
                pool.misses += 1;
                return None;
            }
            pool.free_len -= 1;
            let parked = pool.free[pool.free_len];
            if parked.total >= total {
                return Some(parked.base);
            }
            pool.misses += 1;
            unsafe { crate::jitframe::dealloc_off_gc_block(parked.base, parked.total) };
            None
        })
        .ok()
        .flatten()
}

/// Park a block on the free list. `false` when it must go back to the
/// allocator instead: the owned arm, a full list, or no list.
pub(crate) fn give_back_pooled_block(base: *mut u8, total: usize) -> bool {
    if !jitframe_pool_enabled() {
        return false;
    }
    let slot_result = FRAME_SLOT.try_with(|slot| {
        let state = slot.get();
        if state.base.is_null() {
            slot.set(SlotState {
                base,
                total,
                reaper_armed: state.reaper_armed,
            });
            arm_frame_slot_reaper(slot);
            return Some(ParkedBlock::EMPTY);
        }
        if state.total < total {
            slot.set(SlotState {
                base,
                total,
                reaper_armed: state.reaper_armed,
            });
            return Some(ParkedBlock {
                base: state.base,
                total: state.total,
            });
        }
        None
    });
    match slot_result {
        Ok(Some(displaced)) if displaced.base.is_null() => return true,
        Ok(Some(displaced)) => {
            // The slot kept the larger block. Park the smaller one on the list.
            if give_to_pool_list(displaced.base, displaced.total) {
                return true;
            }
            unsafe { crate::jitframe::dealloc_off_gc_block(displaced.base, displaced.total) };
            return true;
        }
        _ => {}
    }
    give_to_pool_list(base, total)
}

fn give_to_pool_list(base: *mut u8, total: usize) -> bool {
    FRAME_POOL
        .try_with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.free_len >= FRAME_POOL_CAPACITY {
                return false;
            }
            if !pool.reaper_armed {
                pool.reaper_armed = true;
                // First touch registers its destructor for this thread.
                let _ = POOL_REAPER.try_with(|_| ());
            }
            let at = pool.free_len;
            pool.free[at] = ParkedBlock { base, total };
            pool.free_len += 1;
            true
        })
        .unwrap_or(false)
}

/// Tally one owned allocation.
///
/// A thread-local access and a `RefCell` borrow per frame, which the pooled arm
/// pays too but only alongside a free-list pop it was already going to take.
/// This is charged to the OWNED arm alone, and the default arm is not OWNED —
/// so it is a cost of the probe, not of the shipping path. Anyone reading an
/// owned-arm figure should read it as the allocator round trip plus this.
fn count_owned_frame_block() {
    let _ = FRAME_POOL.try_with(|pool| pool.borrow_mut().owned += 1);
}

/// `(owned, pooled takes, pooled misses)` for this thread since it started.
///
/// The witness that an arm selector actually reached the allocation: a timing
/// difference between two arms that allocated the same way is measuring
/// something else.
pub fn jitframe_pool_counts() -> (u64, u64, u64) {
    FRAME_POOL
        .try_with(|pool| {
            let pool = pool.borrow();
            (pool.owned, pool.taken, pool.misses)
        })
        .unwrap_or((0, 0, 0))
}

/// Which arm `alloc_off_gc_jitframe` allocates through.
///
/// `Unseeded` is the state before the first entry reads the selector, not a
/// third strategy: the first read resolves it to one of the other two and
/// stores that back, so no later read can see it again.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum PoolArm {
    Unseeded = 0,
    Owned = 1,
    Pooled = 2,
}

impl PoolArm {
    /// The arm a [`POOL_ARM`] byte names. A byte outside the three
    /// discriminants reads as `Unseeded`, which re-runs the selection rather
    /// than choosing an arm from a value nothing wrote.
    fn from_stored(stored: u8) -> Self {
        match stored {
            x if x == PoolArm::Owned as u8 => PoolArm::Owned,
            x if x == PoolArm::Pooled as u8 => PoolArm::Pooled,
            _ => PoolArm::Unseeded,
        }
    }
}

/// Process-wide, because the thing it selects is a strategy and not a state:
/// a caller flipping arms between two timed batches wants the flip to hold for
/// whichever thread the next entry runs on.
static POOL_ARM: AtomicU8 = AtomicU8::new(PoolArm::Unseeded as u8);

/// Select the frame-allocation arm for subsequent compiled entries.
///
/// Either direction: `false` selects the owned arm on a build whose default
/// is pooled. Overrides `MAJIT_JITFRAME_POOL`, which is only how a
/// harness that cannot call this — a test binary with no hook of its own —
/// picks an arm.
pub fn set_jitframe_pool(on: bool) {
    let arm = if on { PoolArm::Pooled } else { PoolArm::Owned };
    POOL_ARM.store(arm as u8, Ordering::Relaxed);
}

/// Whether the next frame comes off the pool. One relaxed load on the entry
/// path, which BOTH arms pay, so it cancels out of their difference.
#[inline]
pub fn jitframe_pool_enabled() -> bool {
    match PoolArm::from_stored(POOL_ARM.load(Ordering::Relaxed)) {
        PoolArm::Owned => false,
        PoolArm::Pooled => true,
        PoolArm::Unseeded => seed_jitframe_pool_arm(),
    }
}

/// First-touch arm selection: pooled, unless `MAJIT_JITFRAME_POOL` is set to
/// `0`, which is how the owned arm is asked for.
///
/// The env var reads BOTH ways rather than only switching the pool on, because
/// the pool is what a build gets without asking — a harness that wants the
/// allocator round trip has to be able to say so.
#[cold]
#[inline(never)]
fn seed_jitframe_pool_arm() -> bool {
    let on = std::env::var_os("MAJIT_JITFRAME_POOL").is_none_or(|v| v != "0");
    set_jitframe_pool(on);
    on
}

// ── compiled-entry frame-build probe ─────────────────────────────────────
//
// The count lives here, one crate below the backend that reads it and one
// below the metainterp that sets it, because those two do not see each other:
// `majit-metainterp` depends on a backend, never the reverse. The LOOP it
// drives is gated by the reading backend's own feature; this side is a handful
// of atomics that cost nothing until something loads them.

/// Extra frame builds per compiled entry, for splitting what the entry spends
/// before it reaches compiled code.
///
/// The compiled call cannot be repeated — it runs the trace — but its PREFIX
/// can: allocating the jitframe and writing the input arguments into it
/// produces a frame nothing has entered, which is thrown away. That is the only
/// part of the call this can price, and it is the part upstream pays
/// differently (`jitframe_allocate` bump-allocates out of the nursery).
static FRAME_BUILD_REPEATS: AtomicU32 = AtomicU32::new(0);

/// Frame builds the probe actually performed. The witness that an armed count
/// reached the allocation rather than being set on a path nothing ran.
static FRAME_BUILD_PASSES: AtomicU64 = AtomicU64::new(0);

/// Set the extra frame builds per compiled entry, answering what it was.
///
/// Process-wide for the reason [`set_jitframe_pool`] is: it selects a strategy,
/// and a harness flipping arms between two timed batches wants the flip to hold
/// for whichever thread the next entry runs on.
pub fn set_frame_build_repeats(repeats: u32) -> u32 {
    FRAME_BUILD_REPEATS.swap(repeats, Ordering::Relaxed)
}

/// One relaxed load on the entry path, which both arms pay.
#[inline]
pub fn frame_build_repeats() -> u32 {
    FRAME_BUILD_REPEATS.load(Ordering::Relaxed)
}

/// Tally a call's worth of extra frame builds. Once per entry, not once per
/// pass, so the read-modify-write does not scale with the repeat count.
#[inline]
pub fn count_frame_build_passes(passes: u32) {
    if passes != 0 {
        FRAME_BUILD_PASSES.fetch_add(u64::from(passes), Ordering::Relaxed);
    }
}

/// Extra frame builds performed since the process started.
pub fn frame_build_passes() -> u64 {
    FRAME_BUILD_PASSES.load(Ordering::Relaxed)
}

/// Where a held deadframe keeps its jitframe pointer.
///
/// The frame is an ordinary GC object (`jitframe.py` makes JITFRAME a
/// GcStruct and `llmodel.py malloc_jitframe` allocates it like any other),
/// so a collection during the window the deadframe is held promotes it to a
/// different address while every accessor is still reading through the old
/// one. The pointer therefore has to live somewhere the collector rewrites.
///
/// A root slot is that place. `shadowstack.py` (`push_stack` /
/// `pop_stack`) names a root by its POSITION on a per-thread stack rather than
/// by the address of the variable holding it, and `walk_stack_root`
/// (`shadowstack.py`) reads each live slot and stores the forwarded
/// address back into the slot it walked. Addressing by position is what makes
/// the holder's own address irrelevant, so this frame can be returned by value
/// instead of being pinned behind a per-exit heap allocation.
enum JitFrameRoot {
    /// The frame as a GCREF local: `llmodel.py` `return ll_frame` roots the
    /// deadframe on the function's shadow stack (`push_stack` / `pop_stack`).
    /// [`ShadowRoot`] is that slot. A non-LIFO drop nulls the hole
    /// (`walk_stack_root` `if content:`) instead of shrinking past a nested
    /// push.
    Slot(ShadowRoot),
    /// The frame is not a GC object — it was allocated out of the Rust heap
    /// because no type registry was available to allocate it under — so it
    /// cannot move and taking a slot would hand the collector an address
    /// outside both generations.
    Unrooted(GcRef),
}

impl JitFrameRoot {
    #[inline]
    fn get(&self) -> GcRef {
        match self {
            JitFrameRoot::Slot(guard) => guard.get(),
            JitFrameRoot::Unrooted(gcref) => *gcref,
        }
    }
}

/// The descr a compiled exit names: `jf_descr` as an object.
///
/// Owned for a guard exit, whose descr lives in the trace's `FailDescrCell`
/// and is reached through `recover_fail_descr_cell`. Borrowed for a finish
/// exit: that descr is one of the six the metainterp attached to the cpu at
/// `finish_setup` (`compile.py:665 setattr(cpu, name, descr)`), read out of
/// a `CpuDescrCell` copy, and those copies are `'static` — the cell leaks
/// every copy it publishes so that this borrow needs no count. Cloning the
/// `Arc` into every deadframe and dropping it again was measured at 3.2 ns
/// per entry on cel's entry probe, on the arm every finished entry takes.
pub struct ExitDescr {
    ptr: *const dyn majit_ir::Descr,
    owned: bool,
}

// The pointee is an `Arc<dyn Descr>` payload and `Descr: Send + Sync`; the
// raw pointer only records whether this handle holds a count on it.
unsafe impl Send for ExitDescr {}
unsafe impl Sync for ExitDescr {}

impl ExitDescr {
    /// Hold a strong count on `descr`.
    pub fn owned(descr: DescrRef) -> Self {
        ExitDescr {
            ptr: std::sync::Arc::into_raw(descr),
            owned: true,
        }
    }

    /// Point at `descr` without a count. `'static` is the whole argument.
    pub fn borrowed(descr: &'static DescrRef) -> Self {
        ExitDescr {
            ptr: std::sync::Arc::as_ptr(descr),
            owned: false,
        }
    }

    /// `AbstractDescr.show`: `addr` is a [`majit_ir::FailDescrCell`] thin
    /// pointer. The cell keeps the `Arc` alive; this handle does not
    /// bump it.
    ///
    /// # Safety
    /// `addr` must be [`majit_ir::FailDescrCell::thin_ptr`] of a cell that
    /// outlives this handle — a cpu singleton from
    /// [`majit_ir::descr_instance_ptr`], or a guard cell rooted by the
    /// loop's `asmmemmgr_gcreftracers` for as long as the deadframe is read.
    pub unsafe fn from_cell(addr: usize) -> Self {
        let cell = unsafe { majit_ir::FailDescrCell::from_thin_ptr(addr) };
        ExitDescr {
            ptr: std::sync::Arc::as_ptr(&cell.descr),
            owned: false,
        }
    }

    #[inline]
    pub fn get(&self) -> &dyn majit_ir::Descr {
        // Live by construction: owned handles hold a count, borrowed ones
        // point into a `'static`.
        unsafe { &*self.ptr }
    }

    #[inline]
    pub fn as_fail_descr(&self) -> &dyn crate::FailDescr {
        self.get()
            .as_fail_descr()
            .expect("a compiled exit's descr always implements FailDescr")
    }

    /// A strong `Arc` to the descr, whichever way this handle holds it.
    pub fn to_arc(&self) -> DescrRef {
        unsafe {
            std::sync::Arc::increment_strong_count(self.ptr);
            std::sync::Arc::from_raw(self.ptr)
        }
    }
}

impl Clone for ExitDescr {
    fn clone(&self) -> Self {
        if self.owned {
            ExitDescr::owned(self.to_arc())
        } else {
            ExitDescr {
                ptr: self.ptr,
                owned: false,
            }
        }
    }
}

impl Drop for ExitDescr {
    fn drop(&mut self) {
        if self.owned {
            drop(unsafe { std::sync::Arc::from_raw(self.ptr) });
        }
    }
}

impl std::fmt::Debug for ExitDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExitDescr")
            .field("owned", &self.owned)
            .field("descr", &self.get())
            .finish()
    }
}

/// The deadframe of a backend whose frames are jitframes.
pub struct JitFrameDeadFrame {
    /// The JitFrame this deadframe reads through.
    jf_root: JitFrameRoot,
    /// The fail descriptor for this exit.  Stored as `DescrRef`
    /// (`Arc<dyn Descr>`) so the deadframe carries the same Arc identity
    /// the metainterp stamps onto `op.descr` — matching `frame.jf_descr =
    /// descr` (llmodel.py force) line-by-line.
    pub fail_descr: ExitDescr,
    /// Original attached `jf_descr` identity for finish exits emitted by
    /// the metainterp (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub latest_descr: Option<DescrRef>,
    /// Side-channel: caller-prefix layout passed into
    /// `wrap_call_assembler_deadframe_with_caller_prefix` at deadframe
    /// interception.  When `Some`,
    /// the exit's recovery layout is prefixed by this value.  Replaces the old
    /// overlay descr synthesis — the deadframe's `fail_descr` keeps the
    /// callee's own Arc identity rather than being swapped for a synthetic one.
    pub call_assembler_caller_layout: Option<ExitRecoveryLayout>,
    /// Keeps the frame memory alive for non-GC allocations, and decides where
    /// it goes when this deadframe dies. See [`FrameHeapOwner`].
    _heap_owner: Option<FrameHeapOwner>,
}

impl JitFrameDeadFrame {
    /// Take a root slot for `jf_gcref` and hold it for the deadframe's life.
    ///
    /// `heap_owner` decides whether the frame is rooted at all, because it is
    /// the same condition: a frame handed an owner was built on the host heap
    /// precisely because the descr had no `JITFRAME` type id to allocate it
    /// from the nursery under, and a frame allocated from the nursery is
    /// always handed `None`.
    pub fn new(
        jf_gcref: GcRef,
        fail_descr: ExitDescr,
        latest_descr: Option<DescrRef>,
        heap_owner: Option<FrameHeapOwner>,
    ) -> Self {
        let jf_root = if heap_owner.is_some() {
            JitFrameRoot::Unrooted(jf_gcref)
        } else {
            JitFrameRoot::Slot(ShadowRoot::new(jf_gcref))
        };
        JitFrameDeadFrame {
            jf_root,
            fail_descr,
            latest_descr,
            call_assembler_caller_layout: None,
            _heap_owner: heap_owner,
        }
    }

    /// Present a frame allocated by the GC as a rooted deadframe.
    ///
    /// `llmodel.py force` casts the resolved frame to a GCREF and
    /// returns it: the forced frame IS the deadframe, whether it is still
    /// executing or has returned through GUARD_NOT_FORCED_2 / FINISH.
    pub fn borrowing(
        jf_gcref: GcRef,
        fail_descr: ExitDescr,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        JitFrameDeadFrame {
            jf_root: JitFrameRoot::Slot(ShadowRoot::new(jf_gcref)),
            fail_descr,
            latest_descr,
            call_assembler_caller_layout: None,
            _heap_owner: None,
        }
    }

    /// The frame's CURRENT address, re-read through the root slot.
    ///
    /// Not cached anywhere: the slot is the only place the address is correct
    /// after a collection, so every accessor below goes through here.
    #[inline]
    pub fn jf_gcref(&self) -> GcRef {
        self.jf_root.get()
    }

    /// The shadow-stack slot this deadframe already holds.
    ///
    /// `llmodel.py` `return ll_frame`: the deadframe IS the jitframe, and
    /// `ResumeGuardDescr.handle_fail` keeps that object alive for
    /// `blackhole_from_resumedata`. TAGBOX loads re-read this slot
    /// (`cpu.get_int_value(self.deadframe, num)`); they do not acquire a
    /// second one.
    #[inline]
    pub fn shadow_root(&self) -> Option<&ShadowRoot> {
        match &self.jf_root {
            JitFrameRoot::Slot(root) => Some(root),
            JitFrameRoot::Unrooted(_) => None,
        }
    }

    /// `llmodel.py _decode_pos` — translate a logical fail-argument index
    /// through the exit descriptor's recovery locations.
    ///
    /// Cranelift currently publishes a dense identity layout, while dynasm
    /// keeps register spills in their architecture slots. The deadframe is
    /// backend-neutral, so the mapping belongs at this common accessor just
    /// as it does for [`crate::libc_deadframe::LibcJitFrameDeadFrame`].
    #[inline]
    fn slot_of(&self, index: usize) -> Option<usize> {
        let descr = self.fail_descr.as_fail_descr();
        let locs = descr.rd_locs();
        if let Some(&pos) = locs.get(index) {
            (pos != 0xFFFF).then_some(pos as usize)
        } else {
            Some(index)
        }
    }

    #[inline]
    pub fn get_int(&self, index: usize) -> i64 {
        let Some(slot) = self.slot_of(index) else {
            return 0;
        };
        self.get_int_at_slot(slot)
    }

    /// `llmodel.py get_value_direct` — read an already decoded frame slot.
    #[inline]
    pub fn get_int_at_slot(&self, slot: usize) -> i64 {
        // A safe function that dereferences an index the caller chose, so the
        // only thing between an index error and an out-of-bounds read is this
        // check. The frame carries its own slot count in the length word ahead
        // of `jf_frame`, which is the bound. `get_float` and `get_ref` read
        // through here, so one check covers all three.
        let frame = self.jf_gcref();
        let frame_len = unsafe { JitFrame::frame_length(frame.0 as *const JitFrame) };
        if slot >= frame_len as usize {
            return 0;
        }
        unsafe { *((frame.0 + FIRST_ITEM_OFFSET + slot * 8) as *const i64) }
    }

    #[inline]
    pub fn get_float(&self, index: usize) -> f64 {
        f64::from_bits(self.get_int(index) as u64)
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    #[inline]
    pub fn get_savedata_ref(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref().0 + JF_SAVEDATA_OFS as usize) as *const usize) })
    }

    #[inline]
    pub fn try_get_savedata_ref(&self) -> Option<GcRef> {
        let r = self.get_savedata_ref();
        if r.is_null() { None } else { Some(r) }
    }

    #[inline]
    pub fn set_savedata_ref(&mut self, data: GcRef) {
        unsafe { *((self.jf_gcref().0 + JF_SAVEDATA_OFS as usize) as *mut usize) = data.0 };
    }

    /// `cpu.grab_exc_value(deadframe)` (`llmodel.py`): return
    /// `deadframe.jf_guard_exc` without clearing the field.
    #[inline]
    pub fn grab_exc_value(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref().0 + JF_GUARD_EXC_OFS as usize) as *const usize) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FailDescr;
    use crate::finish_descrs::DoneWithThisFrameDescrMulti;
    use crate::jitframe::{alloc_off_gc_jitframe, free_off_gc_jitframe};
    use majit_ir::Type;
    use std::sync::Arc;

    fn a_descr() -> DescrRef {
        Arc::new(majit_ir::descr::SimpleSizeDescr::new(0, 8, 0))
    }

    /// `llmodel.py` returns the JITFRAME itself as the deadframe. Dropping the
    /// host wrapper must not mutate its gcmap: a virtualizable token can keep
    /// that same frame alive and force it later, after the wrapper is gone.
    #[test]
    fn dropping_a_deadframe_preserves_the_frames_gcmap() {
        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(8));
        assert!(!frame.is_null());
        let sentinel = 0x1234usize as *const u8;

        unsafe { (*frame).jf_gcmap = sentinel };
        drop(JitFrameDeadFrame::borrowing(
            GcRef(frame as usize),
            ExitDescr::owned(a_descr()),
            None,
        ));
        assert_eq!(
            unsafe { (*frame).jf_gcmap },
            sentinel,
            "a borrowed frame's map belongs to the call site that pushed it"
        );

        unsafe { (*frame).jf_gcmap = sentinel };
        drop(JitFrameDeadFrame::new(
            GcRef(frame as usize),
            ExitDescr::owned(a_descr()),
            None,
            None,
        ));
        assert_eq!(
            unsafe { (*frame).jf_gcmap },
            sentinel,
            "a returned JITFRAME may still be owned by a virtualizable token"
        );

        unsafe { free_off_gc_jitframe(frame) };
    }

    #[test]
    fn managed_deadframe_decodes_nonidentity_recovery_locations() {
        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(32));
        assert!(!frame.is_null());
        unsafe {
            JitFrame::init(frame, std::ptr::null(), 32);
            crate::llmodel::set_int_value(frame, 2, 22);
            crate::llmodel::set_int_value(frame, 24, 240);
        }

        let descr = Arc::new(DoneWithThisFrameDescrMulti::new(vec![Type::Int, Type::Int]));
        descr.set_rd_locs(vec![24, 2].into());
        let descr_ref: DescrRef = descr;
        let deadframe = JitFrameDeadFrame::new(
            GcRef(frame as usize),
            ExitDescr::owned(descr_ref),
            None,
            None,
        );

        assert_eq!(deadframe.get_int(0), 240);
        assert_eq!(deadframe.get_int(1), 22);
        assert_eq!(deadframe.get_int_at_slot(2), 22);

        drop(deadframe);
        unsafe { free_off_gc_jitframe(frame) };
    }
}

/// Which fixed cost of a compiled entry the probe repeats beside the frame
/// build, selected once per process from `MAJIT_PROBE_EXTRA`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProbeExtraStage {
    None,
    /// `JittedGuard::enter` and its drop.
    Guard,
    /// The jitframe heap selector's thread-local read.
    Heap,
    /// The three `Once`-guarded flag reads.
    Flags,
    /// `cpu_attachments.read()`.
    Attach,
    /// The `jf_descr` word to `(fail_index, descr)`.
    Descr,
    /// One clone and drop of the exit descr's `Arc`.
    Arc,
}

/// `MAJIT_PROBE_EXTRA`, read once.
pub fn probe_extra_stage() -> ProbeExtraStage {
    static STAGE: std::sync::OnceLock<ProbeExtraStage> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| match std::env::var("MAJIT_PROBE_EXTRA").as_deref() {
        Ok("guard") => ProbeExtraStage::Guard,
        Ok("heap") => ProbeExtraStage::Heap,
        Ok("flags") => ProbeExtraStage::Flags,
        Ok("attach") => ProbeExtraStage::Attach,
        Ok("descr") => ProbeExtraStage::Descr,
        Ok("arc") => ProbeExtraStage::Arc,
        _ => ProbeExtraStage::None,
    })
}
