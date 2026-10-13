//! Size-class arena allocator for old-generation objects.
//!
//! Structural port of `rpython/memory/gc/minimarkpage.py`.  Memory is split
//! into arenas, pages, and same-sized blocks.  Free blocks and free pages are
//! chained through their first machine word, exactly like upstream's
//! `llarena` representation; allocations are deliberately not zero-filled.

use std::alloc::{self, Layout};
use std::ptr;

const WORD: usize = std::mem::size_of::<usize>();
const ARENA_ALIGN: usize = if crate::header::GcHeader::ALIGN > WORD {
    crate::header::GcHeader::ALIGN
} else {
    WORD
};
/// Hashed page filter in front of the arena range index. 16 bits, one 8 KB
/// table: a cleared bit is an exact "not in any arena" answer, so a trace
/// slot that names an immortal or a raw block never binary-searches the
/// range list. A set bit still goes through the exact range and live-word
/// test. Over-admission is harmless.
const ARENA_PAGE_FILTER_BITS_LOG2: u32 = 16;
const ARENA_PAGE_FILTER_BITS: usize = 1 << ARENA_PAGE_FILTER_BITS_LOG2;
const ARENA_PAGE_FILTER_WORDS: usize = ARENA_PAGE_FILTER_BITS / 64;
const ARENA_PAGE_BYTES: usize = 4096;
/// Two pages hashed to one slot. The exact range walk decides.
const ARENA_PAGE_OWNER_COLLISION: usize = usize::MAX;

#[repr(C)]
struct ArenaReference {
    base: *mut u8,
    layout: Layout,
    /// Pyre's hybrid heap needs an exact answer to "is this an allocated GC
    /// block?", a question RPython's typed pointers never ask. One bit per
    /// arena word keeps that adaptation on the allocator which owns the
    /// lifetime, without an address-keyed side table.
    live_words: Vec<usize>,
    nfreepages: usize,
    totalpages: usize,
    freepages: *mut u8,
    nextarena: *mut ArenaReference,
}

#[repr(C)]
struct PageHeader {
    nextpage: *mut PageHeader,
    arena: *mut ArenaReference,
    nfree: usize,
    freeblock: *mut u8,
}

/// `minimarkpage.py:ArenaCollection`.
pub struct ArenaCollection {
    pub arena_size: usize,
    pub page_size: usize,
    pub small_request_threshold: usize,
    pub arenas_count: usize,
    page_for_size: Vec<*mut PageHeader>,
    full_page_for_size: Vec<*mut PageHeader>,
    old_page_for_size: Vec<*mut PageHeader>,
    old_full_page_for_size: Vec<*mut PageHeader>,
    nblocks_for_size: Vec<usize>,
    hdrsize: usize,
    max_pages_per_arena: usize,
    arenas_lists: Vec<*mut ArenaReference>,
    old_arenas_lists: Vec<*mut ArenaReference>,
    current_arena: *mut ArenaReference,
    /// pyre-local old-generation membership support.  The arena lists above
    /// retain minimarkpage.py's allocator shape; this flat range index avoids
    /// pointer-chasing every bucket for arbitrary-word validity checks.
    /// It changes only when an arena is allocated or freed.
    arena_ranges: Vec<(usize, usize, *mut ArenaReference)>,
    /// See [`ARENA_PAGE_FILTER_BITS_LOG2`]. Rebuilt when an arena is freed,
    /// because a shared hash bit cannot be cleared. Allocating an arena only
    /// ORs its pages in.
    arena_pages: Box<[u64; ARENA_PAGE_FILTER_WORDS]>,
    /// Arena pointer for the page hash, or 0 if none, or
    /// [`ARENA_PAGE_OWNER_COLLISION`] when two arenas share the slot.
    /// A unique owner answers `contains` without walking `arena_ranges`.
    arena_page_owner: Box<[usize]>,
    /// The range [`contains`] matched last.  Every field store that reaches the
    /// write barrier asks the membership question once, and consecutive stores
    /// name the same arena nearly always, so re-testing the previous answer
    /// first replaces the binary search with two compares on that path.  A hit
    /// can only be stale if the range left `arena_ranges`, so `free_arena`
    /// clears it; inserting leaves every existing range where it was.
    /// `(usize::MAX, 0)` is the empty state — no address satisfies it.
    last_range_hit: std::cell::Cell<(usize, usize, *mut ArenaReference)>,
    /// Covering `[arena_bound_start, arena_bound_end)` of every entry in
    /// `arena_ranges`. [`Self::contains`] rejects an address outside it before
    /// the binary search; an address inside still takes the exact lookup.
    /// `usize::MAX..0` is the empty bound and holds no address. Refreshed
    /// wherever `arena_ranges` gains or loses an arena.
    arena_bound_start: usize,
    arena_bound_end: usize,
    min_empty_nfreepages: usize,
    pub num_uninitialized_pages: usize,
    pub total_memory_used: usize,
    pub peak_memory_used: usize,
    pub total_memory_alloced: usize,
    pub peak_memory_alloced: usize,
    live_objects: usize,
    size_class_with_old_pages: isize,
}

// ArenaCollection owns its arena allocations and is only mutated through
// `&mut self`; the collector itself provides single-threaded access.
unsafe impl Send for ArenaCollection {}

impl ArenaCollection {
    pub fn new(arena_size: usize, page_size: usize, small_request_threshold: usize) -> Self {
        assert_eq!(arena_size % WORD, 0);
        assert_eq!(page_size % WORD, 0);
        assert_eq!(
            page_size % ARENA_ALIGN,
            0,
            "page size must preserve the GC object alignment"
        );
        assert_eq!(small_request_threshold % WORD, 0);
        let length = small_request_threshold / WORD + 1;
        let hdrsize = std::mem::size_of::<PageHeader>();
        assert_eq!(
            hdrsize % ARENA_ALIGN,
            0,
            "PageHeader must preserve the GC object alignment"
        );
        assert!(page_size > hdrsize);
        let mut nblocks_for_size = vec![0; length];
        for (i, nblocks) in nblocks_for_size.iter_mut().enumerate().skip(1) {
            *nblocks = (page_size - hdrsize) / (WORD * i);
            assert!(*nblocks > 0);
        }
        let max_pages_per_arena = arena_size / page_size;
        assert!(max_pages_per_arena > 0);
        Self {
            arena_size,
            page_size,
            small_request_threshold,
            arenas_count: 0,
            page_for_size: vec![ptr::null_mut(); length],
            full_page_for_size: vec![ptr::null_mut(); length],
            old_page_for_size: vec![ptr::null_mut(); length],
            old_full_page_for_size: vec![ptr::null_mut(); length],
            nblocks_for_size,
            hdrsize,
            max_pages_per_arena,
            arenas_lists: vec![ptr::null_mut(); max_pages_per_arena],
            old_arenas_lists: vec![ptr::null_mut(); max_pages_per_arena],
            current_arena: ptr::null_mut(),
            arena_ranges: Vec::new(),
            arena_pages: Box::new([0; ARENA_PAGE_FILTER_WORDS]),
            arena_page_owner: vec![0usize; ARENA_PAGE_FILTER_BITS].into_boxed_slice(),
            last_range_hit: std::cell::Cell::new((usize::MAX, 0, ptr::null_mut())),
            arena_bound_start: usize::MAX,
            arena_bound_end: 0,
            min_empty_nfreepages: max_pages_per_arena,
            num_uninitialized_pages: 0,
            total_memory_used: 0,
            peak_memory_used: 0,
            total_memory_alloced: 0,
            peak_memory_alloced: 0,
            live_objects: 0,
            size_class_with_old_pages: -1,
        }
    }

    /// `llarena.arena_reset(result, sizeof(Address), 0)` after reading the
    /// next-free link stored in the block's first word.
    unsafe fn take_free_link(block: *mut u8) -> *mut u8 {
        let next = unsafe { *(block as *mut *mut u8) };
        unsafe { *(block as *mut usize) = 0 };
        next
    }

    /// `ArenaCollection.malloc`: allocate an uninitialized arena block.
    pub fn malloc(&mut self, size: usize) -> *mut u8 {
        let nsize = size;
        assert!(nsize > 0, "malloc: size is null");
        assert!(
            nsize <= self.small_request_threshold,
            "malloc: size too big"
        );
        assert_eq!(nsize & (WORD - 1), 0, "malloc: size is not aligned");
        self.total_memory_used += nsize;
        self.live_objects += 1;
        let size_class = nsize / WORD;
        let mut page = self.page_for_size[size_class];
        if page.is_null() {
            page = self.allocate_new_page(size_class);
        }
        unsafe {
            let result = (*page).freeblock;
            let freeblock = if (*page).nfree > 0 {
                (*page).nfree -= 1;
                Self::take_free_link(result)
            } else {
                result.add(nsize)
            };
            (*page).freeblock = freeblock;
            let pageaddr = page as *mut u8;
            if freeblock.offset_from(pageaddr) as usize > self.page_size - nsize {
                self.page_for_size[size_class] = (*page).nextpage;
                (*page).nextpage = self.full_page_for_size[size_class];
                self.full_page_for_size[size_class] = page;
            }
            Self::set_block_live((*page).arena, result as usize, true);
            result
        }
    }

    /// `ArenaCollection.allocate_new_page`.
    fn allocate_new_page(&mut self, size_class: usize) -> *mut PageHeader {
        if self.current_arena.is_null() {
            self.allocate_new_arena();
        }
        unsafe {
            let arena = self.current_arena;
            let result = (*arena).freepages;
            let freepages = if (*arena).nfreepages > 0 {
                (*arena).nfreepages -= 1;
                Self::take_free_link(result)
            } else {
                assert!(self.num_uninitialized_pages > 0);
                self.num_uninitialized_pages -= 1;
                if self.num_uninitialized_pages > 0 {
                    result.add(self.page_size)
                } else {
                    ptr::null_mut()
                }
            };
            (*arena).freepages = freepages;
            if freepages.is_null() {
                assert_eq!((*arena).nfreepages, 0);
                (*arena).nextarena = self.arenas_lists[0];
                self.arenas_lists[0] = arena;
                self.current_arena = ptr::null_mut();
            }
            let page = result as *mut PageHeader;
            ptr::write(
                page,
                PageHeader {
                    nextpage: ptr::null_mut(),
                    arena,
                    nfree: 0,
                    freeblock: result.add(self.hdrsize),
                },
            );
            assert!(self.page_for_size[size_class].is_null());
            self.page_for_size[size_class] = page;
            page
        }
    }

    /// `ArenaCollection._pick_next_arena`.
    fn _pick_next_arena(&mut self) -> bool {
        let mut i = self.min_empty_nfreepages;
        while i < self.max_pages_per_arena {
            if !self.arenas_lists[i].is_null() {
                self.current_arena = self.arenas_lists[i];
                unsafe {
                    self.arenas_lists[i] = (*self.current_arena).nextarena;
                }
                return true;
            }
            i += 1;
            self.min_empty_nfreepages = i;
        }
        false
    }

    /// `ArenaCollection.allocate_new_arena`.
    fn allocate_new_arena(&mut self) {
        if self._pick_next_arena() {
            return;
        }
        self._rehash_arenas_lists();
        if self._pick_next_arena() {
            return;
        }
        // The arena stores GcHeader at the beginning of each old-generation
        // block. On wasm32 the header remains u64-aligned even though WORD is
        // four bytes, so pointer-width alignment is insufficient.
        let layout =
            Layout::from_size_align(self.arena_size, ARENA_ALIGN).expect("invalid arena layout");
        let arena_base = unsafe { alloc::alloc(layout) };
        if arena_base.is_null() {
            alloc::handle_alloc_error(layout);
        }
        self.total_memory_alloced += self.arena_size;
        self.peak_memory_alloced = self.peak_memory_alloced.max(self.total_memory_alloced);
        let arena_end = unsafe { arena_base.add(self.arena_size) } as usize;
        let firstpage = start_of_page(arena_base as usize + self.page_size - 1, self.page_size);
        let npages = (arena_end - firstpage) / self.page_size;
        assert!(npages > 0);
        let arena = Box::into_raw(Box::new(ArenaReference {
            base: arena_base,
            layout,
            live_words: vec![0; self.arena_size.div_ceil(WORD * usize::BITS as usize)],
            nfreepages: 0,
            totalpages: npages,
            freepages: firstpage as *mut u8,
            nextarena: ptr::null_mut(),
        }));
        self.num_uninitialized_pages = npages;
        self.current_arena = arena;
        self.arenas_count += 1;
        let range = (arena_base as usize, arena_end, arena);
        let index = self
            .arena_ranges
            .binary_search_by_key(&range.0, |&(start, _, _)| start)
            .unwrap_err();
        self.arena_ranges.insert(index, range);
        self.refresh_arena_bound();
        self.admit_arena_pages(arena_base as usize, arena_end, arena);
    }

    /// Hash of the 4 KB page containing `addr`, spread the way
    /// `AddressHasher::spread` spreads consecutive pages so they do not
    /// land on one short run of the table.
    #[inline]
    fn arena_page_slot(addr: usize) -> (usize, usize, u64) {
        let spread = ((addr >> 12) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let bit = (spread >> (64 - ARENA_PAGE_FILTER_BITS_LOG2)) as usize;
        (bit, bit >> 6, 1u64 << (bit & 63))
    }

    fn admit_arena_pages(&mut self, base: usize, end: usize, arena: *mut ArenaReference) {
        let arena_addr = arena as usize;
        let mut page = base & !(ARENA_PAGE_BYTES - 1);
        while page < end {
            let (bit, word, mask) = Self::arena_page_slot(page);
            self.arena_pages[word] |= mask;
            let cur = self.arena_page_owner[bit];
            if cur == 0 || cur == arena_addr {
                self.arena_page_owner[bit] = arena_addr;
            } else {
                self.arena_page_owner[bit] = ARENA_PAGE_OWNER_COLLISION;
            }
            page = page.saturating_add(ARENA_PAGE_BYTES);
            if page == 0 {
                break;
            }
        }
    }

    fn rebuild_arena_page_filter(&mut self) {
        self.arena_pages.fill(0);
        self.arena_page_owner.fill(0);
        for i in 0..self.arena_ranges.len() {
            let (start, end, arena) = self.arena_ranges[i];
            self.admit_arena_pages(start, end, arena);
        }
    }

    /// `ArenaCollection.mass_free_prepare`.
    pub(crate) fn mass_free_prepare(&mut self) {
        self.peak_memory_used = self.peak_memory_used.max(self.total_memory_used);
        let mut size_class = self.small_request_threshold / WORD;
        self.size_class_with_old_pages = size_class as isize;
        while size_class >= 1 {
            self.old_page_for_size[size_class] = self.page_for_size[size_class];
            self.old_full_page_for_size[size_class] = self.full_page_for_size[size_class];
            self.page_for_size[size_class] = ptr::null_mut();
            self.full_page_for_size[size_class] = ptr::null_mut();
            size_class -= 1;
        }
    }

    /// `ArenaCollection.mass_free_incremental`.
    pub(crate) fn mass_free_incremental(
        &mut self,
        ok_to_free_func: &mut impl FnMut(*mut u8) -> bool,
        mut max_pages: isize,
    ) -> bool {
        let mut size_class = self.size_class_with_old_pages;
        while size_class >= 1 {
            max_pages = self.mass_free_in_pages(size_class as usize, ok_to_free_func, max_pages);
            if max_pages <= 0 {
                self.size_class_with_old_pages = size_class;
                return false;
            }
            size_class -= 1;
        }
        if size_class >= 0 {
            self._rehash_arenas_lists();
            self.size_class_with_old_pages = -1;
        }
        true
    }

    /// `ArenaCollection.mass_free`, the non-incremental STW entry point.
    pub fn mass_free(&mut self, mut ok_to_free_func: impl FnMut(*mut u8) -> bool) {
        self.mass_free_prepare();
        let complete = self.mass_free_incremental(&mut ok_to_free_func, isize::MAX);
        assert!(
            complete,
            "non-incremental mass_free_in_pages returned false"
        );
    }

    /// `ArenaCollection._rehash_arenas_lists`.
    fn _rehash_arenas_lists(&mut self) {
        std::mem::swap(&mut self.old_arenas_lists, &mut self.arenas_lists);
        self.arenas_lists.fill(ptr::null_mut());
        for i in 0..self.max_pages_per_arena {
            let mut arena = self.old_arenas_lists[i];
            self.old_arenas_lists[i] = ptr::null_mut();
            while !arena.is_null() {
                unsafe {
                    let nextarena = (*arena).nextarena;
                    if (*arena).nfreepages == (*arena).totalpages {
                        self.free_arena(arena);
                    } else {
                        let n = (*arena).nfreepages;
                        assert!(n < self.max_pages_per_arena);
                        (*arena).nextarena = self.arenas_lists[n];
                        self.arenas_lists[n] = arena;
                    }
                    arena = nextarena;
                }
            }
        }
        self.min_empty_nfreepages = 1;
    }

    fn mass_free_in_pages(
        &mut self,
        size_class: usize,
        ok_to_free_func: &mut impl FnMut(*mut u8) -> bool,
        mut max_pages: isize,
    ) -> isize {
        let nblocks = self.nblocks_for_size[size_class];
        let block_size = size_class * WORD;
        let mut remaining_partial_pages = self.page_for_size[size_class];
        let mut remaining_full_pages = self.full_page_for_size[size_class];
        for step in 0..2 {
            let mut page = if step == 0 {
                std::mem::replace(
                    &mut self.old_full_page_for_size[size_class],
                    ptr::null_mut(),
                )
            } else {
                std::mem::replace(&mut self.old_page_for_size[size_class], ptr::null_mut())
            };
            while !page.is_null() {
                let surviving = self.walk_page(page, block_size, ok_to_free_func);
                let nextpage = unsafe { (*page).nextpage };
                if surviving == nblocks {
                    debug_assert_eq!(step, 0, "a non-full page became full while freeing");
                    unsafe { (*page).nextpage = remaining_full_pages };
                    remaining_full_pages = page;
                } else if surviving > 0 {
                    unsafe { (*page).nextpage = remaining_partial_pages };
                    remaining_partial_pages = page;
                } else {
                    self.free_page(page);
                }
                max_pages -= 1;
                if max_pages <= 0 {
                    if step == 0 {
                        self.old_full_page_for_size[size_class] = nextpage;
                    } else {
                        self.old_page_for_size[size_class] = nextpage;
                    }
                    self.page_for_size[size_class] = remaining_partial_pages;
                    self.full_page_for_size[size_class] = remaining_full_pages;
                    return 0;
                }
                page = nextpage;
            }
        }
        self.page_for_size[size_class] = remaining_partial_pages;
        self.full_page_for_size[size_class] = remaining_full_pages;
        max_pages
    }

    /// `ArenaCollection.free_page`.
    fn free_page(&mut self, page: *mut PageHeader) {
        unsafe {
            let arena = (*page).arena;
            (*arena).nfreepages += 1;
            let pageaddr = page as *mut u8;
            *(pageaddr as *mut *mut u8) = (*arena).freepages;
            (*arena).freepages = pageaddr;
        }
    }

    /// `ArenaCollection.walk_page`.
    fn walk_page(
        &mut self,
        page: *mut PageHeader,
        block_size: usize,
        ok_to_free_func: &mut impl FnMut(*mut u8) -> bool,
    ) -> usize {
        unsafe {
            let mut freeblock = (*page).freeblock;
            let mut prevfreeblockat: *mut *mut u8 = ptr::addr_of_mut!((*page).freeblock);
            let mut obj = (page as *mut u8).add(self.hdrsize);
            let mut surviving = 0;
            let mut freed = 0;
            let mut skip_free_blocks = (*page).nfree;
            loop {
                if obj == freeblock {
                    if skip_free_blocks == 0 {
                        break;
                    }
                    skip_free_blocks -= 1;
                    prevfreeblockat = obj as *mut *mut u8;
                    freeblock = *prevfreeblockat;
                } else {
                    // minimarkpage.py:541-542 spells this `ll_assert`, which
                    // the C backend emits as `RPyAssert` — nothing at all
                    // unless RPY_ASSERT is defined — so a shipping PyPy runs
                    // no compare here.  `debug_validate_freeblocks` and the
                    // randomized `mass_free` tests keep enforcing it.
                    debug_assert!(
                        (freeblock as usize) > obj as usize,
                        "freeblocks are not ordered"
                    );
                    if ok_to_free_func(obj) {
                        Self::set_block_live((*page).arena, obj as usize, false);
                        *prevfreeblockat = obj;
                        prevfreeblockat = obj as *mut *mut u8;
                        *prevfreeblockat = freeblock;
                        (*page).nfree += 1;
                        freed += 1;
                    } else {
                        surviving += 1;
                    }
                }
                obj = obj.add(block_size);
            }
            self.total_memory_used -= freed * block_size;
            self.live_objects -= freed;
            surviving
        }
    }

    #[cfg(test)]
    pub(crate) fn object_count(&self) -> usize {
        self.live_objects
    }

    pub(crate) fn debug_validate_freeblocks(&self, site: &str) {
        for size_class in 1..self.page_for_size.len() {
            let block_size = size_class * WORD;
            for (kind, mut page) in [
                ("partial", self.page_for_size[size_class]),
                ("full", self.full_page_for_size[size_class]),
                ("old-partial", self.old_page_for_size[size_class]),
                ("old-full", self.old_full_page_for_size[size_class]),
            ] {
                while !page.is_null() {
                    unsafe {
                        let page_start = page as usize;
                        let first = page_start + self.hdrsize;
                        let end = page_start + self.page_size;
                        let mut free = (*page).freeblock as usize;
                        assert!(
                            free >= first && free <= end,
                            "{site}: {kind} size={block_size} page={page_start:#x} free={free:#x} outside [{first:#x}, {end:#x}]"
                        );
                        assert_eq!(
                            (free - first) % block_size,
                            0,
                            "{site}: {kind} size={block_size} page={page_start:#x} free={free:#x} unaligned"
                        );
                        for index in 0..(*page).nfree {
                            let next = *(free as *const usize);
                            assert!(
                                next > free && next <= end,
                                "{site}: {kind} size={block_size} page={page_start:#x} free[{index}]={free:#x} next={next:#x}"
                            );
                            assert_eq!(
                                (next - first) % block_size,
                                0,
                                "{site}: {kind} size={block_size} page={page_start:#x} next={next:#x} unaligned"
                            );
                            free = next;
                        }
                        page = (*page).nextpage;
                    }
                }
            }
        }
    }

    /// Visit every live arena block start (GC header address).
    ///
    /// `live_words` is one bit per arena word set only at `malloc` of a
    /// payload block, so each set bit is an allocated header rather than an
    /// interior word. `debug_check_consistency` walks these so a black object
    /// the current root trace did not reach still gets `_debug_check_object_marking`.
    pub(crate) fn for_each_live_header(&self, mut visit: impl FnMut(usize)) {
        for &(base, end, arena) in &self.arena_ranges {
            let live_words = unsafe { &(*arena).live_words };
            for (word_idx, &bits) in live_words.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let header = base + (word_idx * usize::BITS as usize + bit) * WORD;
                    if header < end {
                        visit(header);
                    }
                }
            }
        }
    }

    /// Recompute the covering bound from `arena_ranges`. Arenas are few, and
    /// this runs only when one is allocated or freed.
    fn refresh_arena_bound(&mut self) {
        // Ranges are sorted by start and every arena has `arena_size` bytes,
        // so the first start and the last end are the covering bounds.
        if let (Some(&(start, _, _)), Some(&(_, end, _))) =
            (self.arena_ranges.first(), self.arena_ranges.last())
        {
            self.arena_bound_start = start;
            self.arena_bound_end = end;
        } else {
            self.arena_bound_start = usize::MAX;
            self.arena_bound_end = 0;
        }
    }

    /// Exact allocated-block membership for the arena allocator.
    ///
    /// RPython never needs this query: the translated pointer type proves that
    /// an edge names a GC object. Pyre's hybrid heap must reject host pointers,
    /// freed blocks, and uninitialized arena bytes before reading a GC header.
    /// The sorted range index finds the arena and its word bitmap distinguishes
    /// the one valid block start from every other address in that range.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        // The covering bound goes first: it is two loads and rejects every
        // address outside the arenas, which is what the immortal-table root
        // walks and the marking of an immortal child ask about. The last-range
        // cache only ever helps an address that is inside.
        if addr < self.arena_bound_start || addr >= self.arena_bound_end {
            return false;
        }
        let (start, end, arena) = self.last_range_hit.get();
        if addr >= start && addr < end {
            return unsafe { Self::block_is_live(arena, addr) };
        }
        // A hit in the hashed page filter is only a candidate: the range
        // index and the live-word bit still decide.
        let (bit, word, mask) = Self::arena_page_slot(addr);
        if self.arena_pages[word] & mask == 0 {
            return false;
        }
        let owner = self.arena_page_owner[bit];
        if owner != 0 && owner != ARENA_PAGE_OWNER_COLLISION {
            let arena = owner as *mut ArenaReference;
            let base = unsafe { (*arena).base as usize };
            let end = base + self.arena_size;
            if addr >= base && addr < end {
                self.last_range_hit.set((base, end, arena));
                return unsafe { Self::block_is_live(arena, addr) };
            }
        }
        // A handful of arenas is the steady state (each arena is 512 KB).
        // Walking them beats `partition_point`: that call does not inline
        // into the marker, so every traced child paid a binary search.
        if self.arena_ranges.len() <= 32 {
            for &(start, end, arena) in &self.arena_ranges {
                if addr >= start && addr < end {
                    self.last_range_hit.set((start, end, arena));
                    return unsafe { Self::block_is_live(arena, addr) };
                }
            }
            return false;
        }
        let index = self
            .arena_ranges
            .partition_point(|&(start, _, _)| start <= addr);
        if index > 0 && addr < self.arena_ranges[index - 1].1 {
            let range = self.arena_ranges[index - 1];
            self.last_range_hit.set(range);
            unsafe { Self::block_is_live(range.2, addr) }
        } else {
            false
        }
    }

    #[inline]
    unsafe fn block_is_live(arena: *mut ArenaReference, addr: usize) -> bool {
        debug_assert!(!arena.is_null());
        let base = unsafe { (*arena).base as usize };
        let offset = addr - base;
        if !offset.is_multiple_of(WORD) {
            return false;
        }
        let bit = offset / WORD;
        let word = bit / usize::BITS as usize;
        let mask = 1usize << (bit % usize::BITS as usize);
        unsafe { (&(*arena).live_words)[word] & mask != 0 }
    }

    #[inline]
    unsafe fn set_block_live(arena: *mut ArenaReference, addr: usize, live: bool) {
        let base = unsafe { (*arena).base as usize };
        let offset = addr - base;
        debug_assert_eq!(offset % WORD, 0);
        let bit = offset / WORD;
        let word = bit / usize::BITS as usize;
        let mask = 1usize << (bit % usize::BITS as usize);
        if live {
            unsafe { (&mut (*arena).live_words)[word] |= mask };
        } else {
            debug_assert!(unsafe { (&(*arena).live_words)[word] & mask != 0 });
            unsafe { (&mut (*arena).live_words)[word] &= !mask };
        }
    }

    unsafe fn free_arena(&mut self, arena: *mut ArenaReference) {
        unsafe {
            let base = (*arena).base as usize;
            if crate::gc_freelist_diag_enabled() {
                let end = base + (*arena).layout.size();
                for size_class in 1..self.page_for_size.len() {
                    for (kind, mut page) in [
                        ("partial", self.page_for_size[size_class]),
                        ("full", self.full_page_for_size[size_class]),
                        ("old-partial", self.old_page_for_size[size_class]),
                        ("old-full", self.old_full_page_for_size[size_class]),
                    ] {
                        while !page.is_null() {
                            let addr = page as usize;
                            assert!(
                                addr < base || addr >= end,
                                "free_arena base={base:#x} nfree={} total={} still owns {kind} size={} page={addr:#x}",
                                (*arena).nfreepages,
                                (*arena).totalpages,
                                size_class * WORD,
                            );
                            page = (*page).nextpage;
                        }
                    }
                }
            }
            let index = self
                .arena_ranges
                .binary_search_by_key(&base, |&(start, _, _)| start)
                .expect("freed arena missing from range index");
            self.arena_ranges.remove(index);
            self.refresh_arena_bound();
            self.last_range_hit.set((usize::MAX, 0, ptr::null_mut()));
            self.rebuild_arena_page_filter();
            alloc::dealloc((*arena).base, (*arena).layout);
            self.total_memory_alloced -= self.arena_size;
            self.arenas_count -= 1;
            drop(Box::from_raw(arena));
        }
    }
}

impl Drop for ArenaCollection {
    fn drop(&mut self) {
        unsafe {
            if !self.current_arena.is_null() {
                self.free_arena(self.current_arena);
                self.current_arena = ptr::null_mut();
            }
            for i in 0..self.arenas_lists.len() {
                let mut arena = self.arenas_lists[i];
                self.arenas_lists[i] = ptr::null_mut();
                while !arena.is_null() {
                    let next = (*arena).nextarena;
                    self.free_arena(arena);
                    arena = next;
                }
            }
        }
    }
}

/// `minimarkpage.py:start_of_page` (translated path).
pub fn start_of_page(addr: usize, page_size: usize) -> usize {
    addr - addr % page_size
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_SIZE: usize = 256;
    const ARENA_SIZE: usize = PAGE_SIZE * 8;
    const THRESHOLD: usize = 9 * WORD;

    #[test]
    fn allocate_and_malloc_mixed_sizes() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let a = ac.malloc(2 * WORD);
        let b = ac.malloc(3 * WORD);
        let c = ac.malloc(2 * WORD);
        assert_ne!(a, c);
        assert!(ac.contains(a as usize));
        assert!(ac.contains(b as usize));
        assert_eq!(ac.total_memory_used, 7 * WORD);
        assert_eq!((a as usize) % WORD, 0);
    }

    #[test]
    fn malloc_reuses_ordered_free_blocks() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let objects: Vec<_> = (0..5).map(|_| ac.malloc(2 * WORD)).collect();
        ac.mass_free(|obj| obj == objects[1] || obj == objects[3]);
        assert_eq!(ac.total_memory_used, 3 * 2 * WORD);
        let reused_first = ac.malloc(2 * WORD);
        assert_eq!(reused_first, objects[1]);
        assert_eq!(unsafe { *(reused_first as *const usize) }, 0);
        assert_eq!(ac.malloc(2 * WORD), objects[3]);
    }

    #[test]
    fn mass_free_partial_and_emptied_pages() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let block_size = 2 * WORD;
        let per_page = (PAGE_SIZE - std::mem::size_of::<PageHeader>()) / block_size;
        let objects: Vec<_> = (0..per_page + 2).map(|_| ac.malloc(block_size)).collect();
        let survivor = objects[0];
        ac.mass_free(|obj| obj != survivor);
        assert_eq!(ac.total_memory_used, block_size);
        assert!(ac.contains(survivor as usize));
        assert!(!ac.contains(objects[1] as usize));
        assert!(!ac.contains(survivor as usize + WORD));
        let reused = ac.malloc(block_size);
        assert_ne!(reused, survivor);
    }

    #[test]
    fn zero_page_incremental_budget_does_not_underflow() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let block_size = 2 * WORD;
        let _objects: Vec<_> = (0..8).map(|_| ac.malloc(block_size)).collect();
        let used_before = ac.total_memory_used;
        ac.mass_free_prepare();
        let mut visited = 0;

        assert!(!ac.mass_free_incremental(
            &mut |_| {
                visited += 1;
                true
            },
            0,
        ));
        assert_eq!(visited, 0);
        assert_eq!(ac.total_memory_used, used_before);

        while !ac.mass_free_incremental(&mut |_| false, 1) {}
        assert_eq!(ac.total_memory_used, used_before);
    }

    #[test]
    fn whole_arena_is_returned_after_mass_free() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let first = ac.malloc(WORD) as usize;
        while !ac.current_arena.is_null() {
            ac.malloc(WORD);
        }
        assert_eq!(ac.arenas_count, 1);
        ac.mass_free(|_| true);
        assert_eq!(ac.total_memory_used, 0);
        assert_eq!(ac.arenas_count, 0);
        assert_eq!(ac.total_memory_alloced, 0);
        assert!(!ac.contains(first));
    }

    #[test]
    fn contains_cache_does_not_survive_the_arena_it_matched() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let first = ac.malloc(WORD) as usize;
        while !ac.current_arena.is_null() {
            ac.malloc(WORD);
        }
        // Prime the single-entry range cache on the arena about to be freed;
        // without the invalidation in `free_arena` this hit would be replayed
        // against returned memory.
        assert!(ac.contains(first));
        ac.mass_free(|_| true);
        assert_eq!(ac.arenas_count, 0);
        assert!(!ac.contains(first));
    }

    #[test]
    fn contains_bound_covers_every_arena_and_rejects_the_outside() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        assert!(!ac.contains(0));
        assert!(!ac.contains(usize::MAX));
        let mut live = vec![ac.malloc(WORD)];
        while ac.arenas_count < 2 {
            live.push(ac.malloc(WORD));
        }
        let first = live[0] as usize;
        let second = *live.last().unwrap() as usize;
        assert_ne!(first, second);
        assert!(ac.contains(first));
        assert!(ac.contains(second));
        assert!(!ac.contains(0));
        assert!(!ac.contains(usize::MAX));
        // Every block dies. A not-yet-full arena stays `current_arena`, so
        // `arenas_count` need not hit zero; neither payload stays live.
        ac.mass_free(|_| true);
        assert!(!ac.contains(first));
        assert!(!ac.contains(second));
        assert!(!ac.contains(0));
    }

    #[test]
    fn randomized_mass_free_matches_live_set() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let mut live = Vec::new();
        let mut state = 0x1234_5678_u64;
        for _round in 0..50 {
            for _ in 0..37 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let size = ((state as usize % 6) + 1) * WORD;
                live.push((ac.malloc(size), size));
            }
            let before = ac.total_memory_used;
            let mut decisions: Vec<_> = live
                .iter()
                .map(|&(ptr, size)| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (ptr, size, (state >> 63) != 0, false)
                })
                .collect();
            ac.mass_free(|obj| {
                let entry = decisions.iter_mut().find(|entry| entry.0 == obj).unwrap();
                assert!(!entry.3);
                entry.3 = true;
                entry.2
            });
            assert!(decisions.iter().all(|entry| entry.3));
            let freed_bytes: usize = decisions
                .iter()
                .filter(|entry| entry.2)
                .map(|entry| entry.1)
                .sum();
            assert_eq!(ac.total_memory_used, before - freed_bytes);
            live = decisions
                .into_iter()
                .filter(|entry| !entry.2)
                .map(|entry| (entry.0, entry.1))
                .collect();
        }
    }

    #[test]
    fn start_of_page_rounds_down() {
        assert_eq!(start_of_page(0x12345, 0x1000), 0x12000);
    }
}
