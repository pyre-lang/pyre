//! Bump-pointer nursery allocator.
//!
//! A fixed-size memory region where young objects are allocated by
//! advancing a free pointer. When the nursery is full, a minor
//! collection copies live objects out.
//!
//! Layout: [header0|payload0|header1|payload1|...|free...top]
//!          ^nursery_start                       ^free  ^top
use std::alloc::{self, Layout};
use std::ptr;

use crate::header::GcHeader;

// ── Heap-allocated nursery pointers ──
//
// incminimark.py:324-325 parity: nursery_free and nursery_top are fields
// in the GC object. The JIT reads their addresses via gc_adr_of_nursery_free
// / gc_adr_of_nursery_top (framework.py:994-997, gc.py:525-531).
//
// In Rust, we allocate these two pointers on the heap via Box<NurseryPtrs>.
// The JIT and the runtime both read/write the SAME memory — no separate
// global statics, no dual-state synchronization.
//
// RPython x86/assembler.py malloc_cond_varsize_frame inline path:
//   ecx = load(nursery_free_adr)
//   edx = ecx + size
//   cmp edx, load(nursery_top_adr)
//   ja slow_path
//   store(nursery_free_adr, edx)

/// incminimark.py:324-325 parity: the two mutable nursery pointers
/// live at a stable heap address so the JIT can hardcode their addresses.
#[repr(C)]
pub struct NurseryPtrs {
    /// incminimark.py:324 self.nursery_free
    pub free: *mut u8,
    /// incminimark.py:325 self.nursery_top
    pub top: *const u8,
}

/// incminimark.py:240 fallback `TRANSLATION_PARAMS["nursery_size"]`.
///
/// This is not the usual runtime nursery size when `read_from_env=True`:
/// incminimark.py:467-470 first asks the environment/cache estimator and
/// only falls back to this value if that fails.
pub const TRANSLATION_NURSERY_SIZE: usize = 896 * 1024;

/// env.py `NURSERY_SIZE_UNKNOWN_CACHE`.
///
/// PyPy's translated incminimark estimates the nursery from cache size and
/// falls back to 4MB.  `estimate_best_nursery_size` implements that probe, so
/// this is the fallback arm of `best_nursery_size_for_l2cache`, taken when the
/// reported L2 is 8MB or smaller — not the default `GcConfig::default` reaches
/// on its own.
pub const DEFAULT_NURSERY_SIZE: usize = 4 * 1024 * 1024;

/// Whether an arena can be made inaccessible on this target.
///
/// `llarena.has_protect` is true for posix and nt and false for everything
/// else, which is what decides whether `post_setup` allocates the rotating
/// nurseries at all.
#[cfg(not(target_arch = "wasm32"))]
pub const HAS_PROTECT: bool = true;
/// See the posix/nt constant above; wasm32 is upstream's `else` arm.
#[cfg(target_arch = "wasm32")]
pub const HAS_PROTECT: bool = false;

/// Alignment of a nursery arena allocation.
///
/// `llimpl_arena_malloc` is a plain `calloc`/`malloc` with no extra alignment.
/// On the 64-bit hosts this crate supports, those return 16-byte aligned
/// blocks — enough for the bump allocator's `MEMORY_ALIGNMENT` objects.
const NURSERY_ARENA_ALIGN: usize = 16;

/// The granularity [`protect_arena`] works in.
#[cfg(not(target_arch = "wasm32"))]
fn page_size() -> usize {
    region::page::size()
}

/// The layout every arena is allocated and freed with.
///
/// `llimpl_arena_malloc` requests exactly `size` bytes; Drop uses this same
/// layout so `dealloc` matches.
fn arena_layout(size: usize) -> Layout {
    Layout::from_size_align(size, NURSERY_ARENA_ALIGN).expect("invalid nursery layout")
}

/// `llimpl_arena_malloc(..., zero=True)` for one nursery-sized arena.
fn alloc_arena(size: usize) -> *mut u8 {
    let layout = arena_layout(size);
    let start = unsafe { alloc::alloc_zeroed(layout) };
    if start.is_null() {
        alloc::handle_alloc_error(layout);
    }
    start
}

/// `llimpl_arena_protect`.
///
/// Only whole pages strictly inside `[start, start+size)` are changed: start
/// is rounded up and end down to [`page_size()`]. `llimpl_arena_protect`
/// hardcodes 4096; the host page size is the granule `mprotect` actually
/// requires. Partial first and last pages stay accessible, so allocator
/// bookkeeping that shares those pages is not taken with the arena.
///
/// A failure is dropped rather than reported, as `llimpl_protect` drops it
/// ("ignore potential errors"): the protection is a debugging aid and a host
/// that refuses it must still run the program.
#[cfg(not(target_arch = "wasm32"))]
fn protect_arena(start: *mut u8, size: usize, inaccessible: bool) {
    let page = page_size();
    let mask = !(page - 1);
    let start_addr = start as usize;
    let prot_start = (start_addr + page - 1) & mask;
    let prot_end = (start_addr + size) & mask;
    if prot_end > prot_start {
        let protection = if inaccessible {
            region::Protection::NONE
        } else {
            region::Protection::READ_WRITE
        };
        let _ =
            unsafe { region::protect(prot_start as *mut u8, prot_end - prot_start, protection) };
    }
}
/// wasm32 has no `arena_protect`; see [`HAS_PROTECT`].
#[cfg(target_arch = "wasm32")]
fn protect_arena(_start: *mut u8, _size: usize, _inaccessible: bool) {}

/// Nursery memory region with bump-pointer allocation.
///
/// incminimark.py:324-325 parity: nursery_free and nursery_top live in
/// a heap-allocated NurseryPtrs struct at a stable address. Both the JIT
/// inline fast path and the runtime slow path read/write the same fields.
pub struct Nursery {
    /// Start of the nursery memory region.
    start: *mut u8,
    /// Total size of the nursery.
    size: usize,
    /// Heap-allocated free/top pointers. The Box ensures stable addresses
    /// that the JIT can hardcode into compiled code.
    ptrs: Box<NurseryPtrs>,
    /// llarena.py mode-3 parity: poison recycled nursery bytes so tests
    /// expose allocation paths that incorrectly rely on zero-filled memory.
    poison_on_reset: bool,
    /// incminimark.py `debug_rotating_nurseries` — spare arenas, each
    /// inaccessible while it waits its turn.
    ///
    /// Empty unless `PYPY_GC_DEBUG` asked for them. Their point is that a
    /// pointer into a retired nursery faults when it is read, instead of being
    /// answered by whichever object was later allocated over it: the reuse is
    /// what hides a missing root, not the staleness.
    rotating: Vec<*mut u8>,
}

// Safety: The nursery owns its memory exclusively and only one thread accesses it.
unsafe impl Send for Nursery {}

impl Nursery {
    /// incminimark.py allocate_nursery parity:
    ///   self.nursery_free = self.nursery
    ///   self.nursery_top = self.nursery + self.nursery_size
    pub fn new(size: usize) -> Self {
        let start = alloc_arena(size);
        let top = unsafe { start.add(size) };
        let ptrs = Box::new(NurseryPtrs { free: start, top });
        let poison_on_reset = crate::gc_nursery_poison_enabled();
        Nursery {
            start,
            size,
            ptrs,
            poison_on_reset,
            rotating: Vec::new(),
        }
    }

    /// incminimark.py `post_setup` — allocate `count` further arenas and
    /// protect them, so [`Self::debug_rotate`] has a ring to draw from.
    ///
    /// Upstream allocates six. The count is a parameter only so a test can ask
    /// for a shorter ring; a host reads it from `PYPY_GC_DEBUG`.
    pub fn install_debug_rotating_nurseries(&mut self, count: usize) {
        if !HAS_PROTECT {
            return;
        }
        for _ in 0..count {
            let arena = alloc_arena(self.size);
            protect_arena(arena, self.size, true);
            self.rotating.push(arena);
        }
    }

    /// incminimark.py `debug_rotate_nursery`.
    ///
    /// Retire the current arena to the back of the ring — inaccessible — and
    /// take the one at the front. Reports whether a ring was installed.
    ///
    /// The caller owns the precondition upstream states by calling this only
    /// where `nursery_barriers` is still empty: nothing may be living in the
    /// retired arena, because reading it now faults.
    pub fn debug_rotate(&mut self) -> bool {
        if self.rotating.is_empty() {
            return false;
        }
        let old = self.start;
        protect_arena(old, self.size, true);
        let new = self.rotating.remove(0);
        self.rotating.push(old);
        protect_arena(new, self.size, false);
        self.start = new;
        // `debug_rotate_nursery` sets `nursery` and `nursery_top`, and the
        // `nursery_free = nursery` its caller performs at the end of
        // `_minor_collection` lands on the arena installed here.
        self.ptrs.free = new;
        self.ptrs.top = unsafe { new.add(self.size) };
        true
    }

    /// How many spare arenas the rotation ring holds.
    pub fn debug_rotating_nurseries(&self) -> usize {
        self.rotating.len()
    }

    /// `addr` falls in a nursery arena [`Self::debug_rotate`] has retired.
    ///
    /// The current arena is not in `rotating`. Those arenas are protected, so
    /// a store check must use this range and must not read the header.
    pub fn contains_retired(&self, addr: usize) -> bool {
        let size = self.size;
        self.rotating.iter().any(|&arena| {
            let start = arena as usize;
            addr >= start && addr < start + size
        })
    }

    /// `_minor_collection` under `gc_nursery_debug` resets the recycled range
    /// in `arena_reset` mode 3, the one that fills it with garbage.
    ///
    /// Additive because the arena reads `MAJIT_GC_NURSERY_POISON` for itself:
    /// either spelling selects the mode, and neither turns the other off.
    pub fn set_nursery_debug(&mut self, on: bool) {
        self.poison_on_reset |= on;
    }

    /// incminimark.py malloc_fixedsize parity:
    ///   result = self.nursery_free
    ///   self.nursery_free = new_free = result + totalsize
    ///   if new_free > self.nursery_top: collect_and_reserve()
    ///
    /// Returns null when the nursery is full (caller must collect & retry).
    #[inline]
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        // Ensure minimum size for forwarding during collection.
        let total_size = total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE);
        let total_size = (total_size + crate::header::MEMORY_ALIGNMENT - 1)
            & !(crate::header::MEMORY_ALIGNMENT - 1);

        let result = self.ptrs.free;
        // Use wrapping_add for the bound probe: when the nursery is already
        // full `result` sits at `top` (one-past-end), so an in-bounds `.add`
        // would be UB. RPython's integer arithmetic cannot hit this because
        // it operates on untyped addresses.
        let new_free = result.wrapping_add(total_size);
        if new_free as *const u8 > self.ptrs.top {
            return ptr::null_mut();
        }
        self.ptrs.free = new_free;
        result
    }

    /// incminimark.py:1946 parity: reset nursery after minor collection.
    ///   self.nursery_free = self.nursery
    ///
    /// incminimark.py/1938 parity: `malloc_zero_filled = False` and
    /// `arena_reset(..., 0)` leave recycled bytes untouched; allocation
    /// sites initialize their own GC-pointer fields.  Poison mode mirrors
    /// llarena.py mode 3 for detecting violations of that contract.
    ///
    /// Allocation sites initialize their own payload. The wasm backend
    /// skips `GcRewriterImpl.clear_gc_fields`, so its `New*` / `NewArray*`
    /// bumps and `wasm_jit_alloc*` helpers emit `memory.fill` / `write_bytes`
    /// themselves. Recycled nursery bytes stay dirty, matching native.
    pub fn reset(&mut self) {
        self.reset_range(self.start as usize, self.start as usize + self.size);
        self.ptrs.free = self.start;
    }

    /// Reset one free range while leaving pinned-object bytes intact.
    ///
    /// `IncrementalMiniMarkGC._minor_collection` calls `arena_reset` once for
    /// every gap between surviving pinned objects. Poison mode fills the gap
    /// with 0xAA; otherwise the bytes stay as the last occupant left them.
    pub fn reset_range(&mut self, start: usize, end: usize) {
        debug_assert!(start >= self.start as usize);
        debug_assert!(start <= end);
        debug_assert!(end <= self.start as usize + self.size);
        // This is a safe fn that writes raw bytes at caller-supplied
        // addresses, so the bounds have to hold in release too: an
        // out-of-range end would write outside the arena, and `start > end`
        // would wrap the length into a near-`usize::MAX` fill. Intersect with
        // the arena instead of trusting the caller.
        let lo = start.max(self.start as usize);
        let hi = end.min(self.start as usize + self.size);
        if lo >= hi {
            return;
        }
        let (start, len) = (lo, hi - lo);
        if self.poison_on_reset {
            unsafe {
                ptr::write_bytes(start as *mut u8, 0xAA, len);
            }
        }
    }

    /// incminimark.py:676: current nursery_free.
    #[inline]
    pub fn free_ptr(&self) -> *mut u8 {
        self.ptrs.free
    }

    /// gc.py get_nursery_free_addr parity.
    #[inline]
    pub fn free_addr(&self) -> usize {
        std::ptr::addr_of!(self.ptrs.free) as usize
    }

    /// Set the free pointer (used after collection with pinned objects).
    ///
    /// # Safety
    /// `ptr` must be within the nursery bounds.
    pub unsafe fn set_free_ptr(&mut self, ptr: *mut u8) {
        debug_assert!(ptr as usize >= self.start as usize);
        debug_assert!(ptr as usize <= self.ptrs.top as usize);
        self.ptrs.free = ptr;
    }

    /// incminimark.py:910,1947: set nursery_top (pinned object barriers).
    ///
    /// # Safety
    /// `ptr` must be within the nursery bounds.
    pub unsafe fn set_top_ptr(&mut self, ptr: *const u8) {
        debug_assert!(ptr as usize >= self.start as usize);
        debug_assert!(ptr as usize <= self.start as usize + self.size);
        self.ptrs.top = ptr;
    }

    /// incminimark.py:325: current nursery_top.
    #[inline]
    pub fn top_ptr(&self) -> *const u8 {
        self.ptrs.top
    }

    /// gc.py get_nursery_top_addr parity.
    #[inline]
    pub fn top_addr(&self) -> usize {
        std::ptr::addr_of!(self.ptrs.top) as usize
    }

    /// Start of nursery.
    #[inline]
    pub fn start_ptr(&self) -> *const u8 {
        self.start
    }

    /// Total nursery size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bytes currently used.
    #[inline]
    pub fn used(&self) -> usize {
        self.ptrs.free as usize - self.start as usize
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.ptrs.top as usize - self.ptrs.free as usize
    }

    /// Check if an address is within the nursery.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.start as usize && addr < (self.start as usize + self.size)
    }

    /// Whether recycled nursery bytes are filled with the llarena-style
    /// uninitialized-memory poison.  This is cached at nursery construction so
    /// traced-slot checks do not consult the environment during collection.
    #[inline]
    pub(crate) fn poison_enabled(&self) -> bool {
        self.poison_on_reset
    }
}

impl Drop for Nursery {
    fn drop(&mut self) {
        let layout = arena_layout(self.size);
        // Unprotect a parked arena before handing it back: the allocator
        // writes its own bookkeeping into the block it reclaims, and an
        // inaccessible one would fault inside `dealloc`.
        for &arena in &self.rotating {
            protect_arena(arena, self.size, false);
        }
        unsafe {
            for &arena in &self.rotating {
                alloc::dealloc(arena, layout);
            }
            alloc::dealloc(self.start, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `llimpl_arena_protect` only covers whole pages strictly inside the
    /// arena, so a retired arena's partial edge pages stay accessible.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn a_retired_arena_protects_interior_pages_only() {
        if !HAS_PROTECT {
            return;
        }
        let page = page_size();
        let size = 4 * page + 123;
        let mut nursery = Nursery::new(size);
        nursery.install_debug_rotating_nurseries(1);
        let retired = nursery.start_ptr() as usize;
        assert!(nursery.debug_rotate());

        let mask = !(page - 1);
        let round_up = (retired + page - 1) & mask;
        let first_page = retired & mask;
        let last_page = (retired + size - 1) & mask;
        let wholly_inside =
            |page_start: usize| page_start >= retired && page_start + page <= retired + size;

        let first = region::query(retired as *const u8).expect("first byte mapped");
        if wholly_inside(first_page) {
            assert_eq!(first.protection(), region::Protection::NONE);
        } else {
            assert!(first.is_readable());
        }

        let last = region::query((retired + size - 1) as *const u8).expect("last byte mapped");
        if wholly_inside(last_page) {
            assert_eq!(last.protection(), region::Protection::NONE);
        } else {
            assert!(last.is_readable());
        }

        let interior = region::query(round_up as *const u8).expect("interior page mapped");
        assert_eq!(interior.protection(), region::Protection::NONE);
    }

    /// `debug_rotate_nursery` hands out the ring's front arena and sends the
    /// retired one to the back, so the count never changes and no address is
    /// reused until the whole ring has turned.
    #[test]
    fn rotating_hands_out_a_fresh_arena_and_keeps_the_ring_full() {
        if !HAS_PROTECT {
            return;
        }
        let mut nursery = Nursery::new(4096);
        nursery.install_debug_rotating_nurseries(2);
        assert_eq!(nursery.debug_rotating_nurseries(), 2);

        // The JIT hardcodes these two addresses, so a rotation that moved them
        // would leave compiled code bumping a pointer pair nothing reads.
        let free_slot = nursery.free_addr();
        let top_slot = nursery.top_addr();

        let first = nursery.start_ptr() as usize;
        assert!(nursery.debug_rotate());
        let second = nursery.start_ptr() as usize;
        assert_ne!(second, first, "a rotation must hand out a different arena");
        assert_eq!(
            nursery.debug_rotating_nurseries(),
            2,
            "the retired arena takes the place of the one taken"
        );
        assert_eq!(
            nursery.free_ptr() as usize,
            second,
            "the bump pointer follows the arena"
        );
        assert_eq!(nursery.top_ptr() as usize, second + nursery.size());
        assert_eq!(nursery.free_addr(), free_slot, "the pointer pair stays put");
        assert_eq!(nursery.top_addr(), top_slot);

        assert!(nursery.debug_rotate());
        let third = nursery.start_ptr() as usize;
        assert_ne!(third, second);
        assert_ne!(
            third, first,
            "two spares means three arenas before a repeat"
        );

        assert!(nursery.debug_rotate());
        assert_eq!(
            nursery.start_ptr() as usize,
            first,
            "and then the ring comes round"
        );
    }

    /// `debug_rotate_nursery` opens with `if self.debug_rotating_nurseries:` —
    /// without `PYPY_GC_DEBUG` there is nothing to rotate to and the arena
    /// stays.
    #[test]
    fn rotating_without_a_ring_leaves_the_arena_alone() {
        let mut nursery = Nursery::new(4096);
        let before = nursery.start_ptr() as usize;
        assert!(!nursery.debug_rotate());
        assert_eq!(nursery.start_ptr() as usize, before);
    }

    #[test]
    fn test_nursery_create() {
        let nursery = Nursery::new(4096);
        assert_eq!(nursery.size(), 4096);
        assert_eq!(nursery.used(), 0);
        assert_eq!(nursery.remaining(), 4096);
    }

    #[test]
    fn test_nursery_alloc() {
        let mut nursery = Nursery::new(4096);

        let p1 = nursery.alloc(32);
        assert!(!p1.is_null());
        // At least MIN_NURSERY_OBJ_SIZE, aligned to 8
        assert!(nursery.used() >= 32);

        let p2 = nursery.alloc(64);
        assert!(!p2.is_null());
        assert!(p2 > p1);
    }

    #[test]
    fn test_nursery_full() {
        let mut nursery = Nursery::new(64);

        let p1 = nursery.alloc(32);
        assert!(!p1.is_null());

        // The nursery is 64 bytes; after allocating 32 (aligned up to at least 16),
        // we may or may not have room for another 32.
        // But asking for the full size should eventually fail.
        let p_big = nursery.alloc(64);
        // Should fail: not enough space
        assert!(p_big.is_null());
    }

    #[test]
    fn test_nursery_reset() {
        let mut nursery = Nursery::new(4096);

        nursery.alloc(100);
        nursery.alloc(100);
        assert!(nursery.used() > 0);

        nursery.reset();
        assert_eq!(nursery.used(), 0);
        assert_eq!(nursery.remaining(), 4096);
    }

    #[test]
    fn test_nursery_contains() {
        let nursery = Nursery::new(4096);
        let start = nursery.start_ptr() as usize;
        let top = nursery.top_ptr() as usize;

        assert!(nursery.contains(start));
        assert!(nursery.contains(start + 100));
        assert!(!nursery.contains(top));
        assert!(!nursery.contains(start.wrapping_sub(1)));
    }

    #[test]
    fn test_nursery_zero_filled() {
        let mut nursery = Nursery::new(4096);
        let p = nursery.alloc(64);
        assert!(!p.is_null());
        // Check memory is zero-filled
        for i in 0..64 {
            assert_eq!(unsafe { *p.add(i) }, 0);
        }
    }

    #[test]
    fn test_nursery_alignment() {
        let mut nursery = Nursery::new(4096);
        for _ in 0..10 {
            let p = nursery.alloc(17); // unaligned request
            assert!(!p.is_null());
            assert_eq!(p as usize % 8, 0); // result is 8-byte aligned
        }
    }

    #[test]
    fn test_nursery_instance_addrs_stable() {
        // gc.py:525-531 parity: nursery free/top addresses are per-instance
        // fields of the live GC descriptor, not a process-global singleton.
        let nursery = Nursery::new(4096);
        let free_addr = nursery.free_addr();
        let top_addr = nursery.top_addr();
        assert_ne!(free_addr, 0);
        assert_ne!(top_addr, 0);
        assert_ne!(free_addr, top_addr);
        let free_val = unsafe { *(free_addr as *const *mut u8) };
        assert_eq!(free_val, nursery.free_ptr());
    }
}
