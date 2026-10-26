pub use collector::{FinalizerDeque, FinalizerQueueTable, HEAP_DUMP_EIO};
pub use gcreftracer::{GcTable, install_gc_table_walker};
pub use header::{
    GC_FLAG_SHIFT, GC_HEADER_WORD_SIZE, GC_IMMORTAL_LEAF_HEADER, GC_TYPE_ID_BITS,
    GCFLAG_NO_HEAP_PTRS, GCFLAG_TRACK_YOUNG_PTRS, GCREF, GCREFOpaque, GcType,
};

/// `history.py` `ConstPtr` is rooted by the holder that stores the box
/// (`trace_index`), not by every slot `intern` has ever recorded.
/// The registration stays so a collector still has a named walker; it
/// does not keep dead trace constants alive. Compiled-code constants
/// use `gcreftracer.GcTable`.
/// Idempotent: `register_extra_root_walker` dedups by function address.
pub fn install_const_ptr_table_walker() {
    shadow_stack::register_extra_root_walker(const_ptr_table_walker, "const_ptr_table");
    // `gcref_diag` is a P92 intern/ConstPtr hook. Upstream has no such
    // diagnostic: `history.py` `ConstPtr.getref_base` is a field read.
    // `diag_stale_gcref` returns immediately unless `diag_p92_trace_io`
    // is on, so installing it unconditionally was an acquire load plus
    // an indirect call on every intern. Install only when the knob is on.
    if diag_p92_enabled() {
        majit_ir::set_gcref_diag(Some(diag_stale_gcref));
    }
    majit_ir::set_gc_current_object_address(Some(gc_current_object_address));
}

fn const_ptr_table_walker(_visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {}

/// Temporary P92 diagnostic. `PYRE_DIAG_P92` classifies a GC word at intern /
/// ConstPtr construction / recorder decode: nursery, forwarded, header valid.
/// Panics with a backtrace at the first stale address. Remove after the fix.
///
/// 0 uninit, 1 absent, 2 present. Same load as `have_debug_prints` /
/// `OP_HAVE_DEBUG_PRINTS` (`pypy_have_debug_prints & 1`). Upstream has
/// no such knob, so the off path is one relaxed load.
static DIAG_P92: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[inline(always)]
pub fn diag_p92_enabled() -> bool {
    match DIAG_P92.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => diag_p92_enabled_init(),
    }
}

#[cold]
#[inline(never)]
fn diag_p92_enabled_init() -> bool {
    let on = std::env::var_os("PYRE_DIAG_P92").is_some();
    DIAG_P92.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
    on
}

/// Blackhole/intern eprints. Off: those I/O paths Heisenbug the 6-box numbering.
pub fn diag_p92_trace_io() -> bool {
    false
}

pub fn diag_stale_gcref(addr: usize, site: &'static str) {
    if !diag_p92_trace_io() || addr == 0 {
        return;
    }
    if !gc_sync::is_initialized() {
        return;
    }
    gc_sync::gc_query_reentrant(|gc| gc.diag_stale_gcref(addr, site));
}

/// GC traits and interfaces for the JIT.
///
/// The GC subsystem provides:
/// 1. Object allocation (nursery bump-pointer + old gen)
/// 2. Write barrier insertion
/// 3. GC-aware IR rewriting (NEW_* → inline nursery allocation)
/// 4. Stack maps for compiled code
///
/// Reference: rpython/memory/gc/incminimark.py, rpython/jit/backend/llsupport/gc.py
use majit_ir::{Const, ConstMap, GcRef, OpRc};
pub use nursery_clock::HostNurseryClock;
pub use trace::{
    ClassTypeLayout, CustomDataLayout, TypeEntry, TypeEntryTail, TypeInfo, TypeInfoLayout,
    VarSizeTypeInfoLayout,
};

/// Runtime layout of one registered variable-size GC object.
///
/// These are the facts `rgc.ll_arraymove` gets from the array lltype in
/// RPython. The residual target retains upstream's four-argument ABI and
/// recovers them from the collector's existing TYPE_INFO row instead of a
/// pyre-only extra argument or a parallel layout table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GcVarSizeLayout {
    pub base_size: usize,
    pub item_size: usize,
    pub items_have_gc_ptrs: bool,
}

pub mod address_dict;
pub mod collector;
pub mod gc_sync;
pub mod gcreftracer;
pub mod header;
pub mod hook;
pub mod hook_cell;
pub mod minimarkpage;
pub mod nursery;
pub mod nursery_clock;
pub mod oldgen;
mod process_ceiling;
pub mod rawrefcount;
pub mod rewrite;
pub mod rgil;
#[cfg(not(target_arch = "wasm32"))]
pub mod rmmap;
pub mod shadow_stack;
pub mod trace;
pub mod weakref;

static AFTER_MINOR_COLLECTION_FN: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

// Runtime type ids for RPython's varsize `rstr.STR` / `rstr.UNICODE` leaves.
// The object model publishes them after `TypeInfo` registration; both JIT
// backends read the same ids when constructing their builtin string
// ArrayDescrs, so interpreter, blackhole, and compiled allocations share one
// collector identity.
static LOWLEVEL_STR_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LOWLEVEL_UNICODE_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

pub fn set_lowlevel_str_type_id(id: u32) {
    debug_assert_ne!(id, 0);
    LOWLEVEL_STR_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
}

/// Pair of [`set_lowlevel_str_type_id`] for a test that published an id only
/// for the duration of one call. `0` is the unpublished sentinel.
pub fn clear_lowlevel_str_type_id() {
    LOWLEVEL_STR_TYPE_ID.store(0, std::sync::atomic::Ordering::Release);
}

pub fn set_lowlevel_unicode_type_id(id: u32) {
    debug_assert_ne!(id, 0);
    LOWLEVEL_UNICODE_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
}

pub fn lowlevel_str_type_id() -> u32 {
    LOWLEVEL_STR_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

pub fn lowlevel_unicode_type_id() -> u32 {
    LOWLEVEL_UNICODE_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

/// Register the callback installed as `finished_minor_collection` by
/// framework.py:135-138. Called once when the JIT counter is initialized.
pub fn register_after_minor_collection_hook(f: fn()) {
    let _ = AFTER_MINOR_COLLECTION_FN.set(f);
}

pub(crate) fn invoke_after_minor_collection_hook() {
    if let Some(f) = AFTER_MINOR_COLLECTION_FN.get() {
        f();
    }
}

bitflags::bitflags! {
    /// The `GCFLAG_*` bits an object header carries.
    ///
    /// incminimark.py declares each as `first_gcflag << N`, with
    /// `first_gcflag = 1 << (LONG_BIT//2)`; the position here is N and
    /// `GcHeader` applies `FLAG_SHIFT` when it writes the word, whose lower
    /// half is the type id.  `every_gcflag_is_the_bit_incminimark_gives_it`
    /// reads the positions back out of `incminimark.py`.
    ///
    /// The names are incminimark's own, so a flag test and the RPython line it
    /// came from read alike: a constant carrying the `GCFLAG_` prefix is one
    /// incminimark declares, at the position it declares, and the two without
    /// it are this GC's own.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct GcFlags: u64 {
        /// Something must be done to track the young pointers this object may
        /// hold. Not set on young objects -- any young object may point to any
        /// other -- except large arrays, where `GCFLAG_HAS_CARDS` does the
        /// tracking and this is set anyway to shorten the write barrier. On old
        /// and prebuilt objects it is usually set, and writing a pointer clears
        /// it.
        const GCFLAG_TRACK_YOUNG_PTRS = majit_ir::GCFLAG_TRACK_YOUNG_PTRS;
        /// Set on a prebuilt object unless it is already listed in
        /// `prebuilt_root_objects`.
        ///
        /// A state, not a claim about the payload's shape. While it is set the
        /// object holds no heap pointer, and the write barrier keeps that true:
        /// the first pointer written into the object clears this flag and
        /// appends the object to `prebuilt_root_objects`. That is what lets
        /// marking skip an object still carrying it
        /// (`IncrementalMiniMarkGC.visit`) and lets `_rrc_major_free` read it as
        /// "immortal, never traced so far".
        ///
        /// The invariant holds only where *every* pointer store into such an
        /// object runs the barrier. Upstream gets that from the GC transform; a
        /// hand-written store that skips it leaves a heap pointer behind a set
        /// flag, and marking then walks past a live object.
        const GCFLAG_NO_HEAP_PTRS = majit_ir::GCFLAG_NO_HEAP_PTRS;
        /// The object survived a major collection.
        const GCFLAG_VISITED = 1 << 2;
        /// Set on a nursery object whose id or identityhash was asked for:
        /// space the size of the object is already reserved in the non-movable
        /// part.
        const GCFLAG_HAS_SHADOW = 1 << 3;
        /// Set temporarily during a major collection to order finalizers.
        const GCFLAG_FINALIZATION_ORDERING = 1 << 4;
        /// Reserved for RPython.
        const GCFLAG_EXTRA = 1 << 5;
        /// Set on an externally raw-malloced array of pointers: it carries
        /// extra space in front for a bitfield, one bit per
        /// `card_page_indices` indices.
        const GCFLAG_HAS_CARDS = 1 << 6;
        /// At least one card bit is set. This is the most significant bit of
        /// the byte holding `GCFLAG_TRACK_YOUNG_PTRS`, which the x86 backend
        /// requires.
        const GCFLAG_CARDS_SET = 1 << 7;
        /// Set on a raw-malloced young object that survived a minor collection.
        const GCFLAG_VISITED_RMY = 1 << 8;
        /// Keep this nursery object in the nursery: a young object carrying it
        /// is not moved out by a minor collection. See `pin()` / `unpin()`.
        const GCFLAG_PINNED = 1 << 9;
        /// The same bit as [`GcFlags::GCFLAG_PINNED`], reused on objects
        /// outside the nursery, where pinning cannot apply. Set means the
        /// object is already an element of `old_objects_pointing_to_pinned`
        /// and need not be added again.
        const GCFLAG_PINNED_OBJECT_PARENT_KNOWN = Self::GCFLAG_PINNED.bits();
        /// `ignore_finalizer()` has been called on the object.
        const GCFLAG_IGNORE_FINALIZER = 1 << 10;
        /// A shadow object whose memory was initialized as it was created, so
        /// tracing out needs no additional copy.
        const GCFLAG_SHADOW_INITIALIZED = 1 << 11;
        /// The `ll_dummy_value` from `rpython.rtyper.rmodel`.
        const GCFLAG_DUMMY = 1 << 12;
        /// The object is already on a finalizer queue.
        ///
        /// Not an incminimark flag -- bit 13 is `_GCFLAG_FIRST_UNUSED`
        /// (`_GCFLAG_FIRST_UNUSED`) there, so this claims the first free bit and
        /// disturbs no RPython position.  incminimark keeps the registered set
        /// purely in its two deques and never asks the question, because
        /// `register_finalizer` is contracted to be called at most once per
        /// object.  That contract is checked only untranslated
        /// (`FinalizerQueue._untranslated_register_finalizer`,
        /// `assert not self._already_registered(obj)`); translated, a second
        /// registration silently appends a second deque entry and the finalizer
        /// runs again on the following major collection.  This flag lets the
        /// queue enforce the contract for callers that cannot establish single
        /// registration statically.
        const FINALIZER_REGISTERED = 1 << 13;
        /// A young raw-malloced object, i.e. one `external_malloc(...,
        /// alloc_young=True)` produced and `free_young_rawmalloced_objects` has
        /// not yet promoted or freed.
        ///
        /// Not an incminimark flag: upstream answers the same question with
        /// `self.young_rawmalloced_objects`, an address dict it also iterates to
        /// free the generation.  pyre needs the `Layout` `alloc::dealloc` wants,
        /// so the iteration half is a `Vec<RawMallocedObject>` and the
        /// membership half had to be a second structure -- an `AddressSet` whose
        /// insert every young rawmalloc birth paid, which on a call-heavy
        /// interpreted loop is every Python frame.  The state belongs on the
        /// object, the way `GCFLAG_VISITED_RMY` keeps the survivor bit for the
        /// same generation in the same function.  Bit 13 is
        /// `_GCFLAG_FIRST_UNUSED` upstream and 13/14 are already claimed, so
        /// this takes the next free position and disturbs no RPython one.
        ///
        /// The name carries no `GCFLAG_` prefix for the reason
        /// `every_gcflag_is_the_bit_incminimark_gives_it` enforces: the prefix
        /// is the claim that incminimark declares the flag, and this GC's own
        /// flags are spelled without it.
        ///
        /// It is also the last position the header admits on wasm32:
        /// `header.rs` gives flags `usize::BITS / 2` bits, so 16 there, and
        /// `GcHeader::with_flags` asserts the fit.  A further flag needs a wider
        /// wasm32 header word, not a higher bit.
        const YOUNG_RAWMALLOC = 1 << 15;
        /// The object's finalizer has already run.
        ///
        /// Not an incminimark flag either -- the queue delivers an object once
        /// and incminimark never asks the question afterwards.  `gc.is_finalized`
        /// does ask it: an object its own finalizer resurrected reports
        /// `_PyGC_FINALIZED`, and with nothing recorded the answer could only be
        /// a constant `false`.
        const FINALIZER_RUN = 1 << 14;
    }
}

/// Low-level trigger stored in an RPython finalizer handler.  It must only
/// schedule app-level work; finalizers themselves run after collection.
pub type FinalizerTriggerFn = fn();

/// The `'FinalizerQueue TAG'` constant of `framework.py`.
///
/// `fq` is the address of the queue's `'static` instance, the identity
/// `finalizer_queue_indexes` keys by. `trigger` is its `ll_finalizer_trigger`.
#[derive(Clone, Copy)]
pub struct FinalizerQueueTag {
    pub fq: usize,
    pub trigger: FinalizerTriggerFn,
}

/// True when the `gc_stress` test feature is compiled in: every allocation
/// may then run a full collection inside `alloc_with_type`, so JIT fast
/// paths that bypass it (inline nursery bump) must stay disabled or the
/// stress coverage silently shrinks to non-JIT allocations.
pub fn gc_stress_enabled() -> bool {
    cfg!(feature = "gc_stress")
}

/// `MAJIT_GC_LIFETIME_LOG` — trace remembered-set adds and old-gen frees.
///
/// Read once.  The gate sits in the write barrier and the old-gen sweep, and
/// `std::env::var_os` takes the environment lock and scans it linearly on every
/// call, so asking per event costs whether or not the variable is set.  Same
/// shape as `majit_metainterp::majit_log_enabled`.
pub fn gc_lifetime_log_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_GC_LIFETIME_LOG").is_some());
    *ENABLED
}

/// `MAJIT_GC_FREELIST_DIAG` — dump the intra-page free lists around an arena
/// release.
///
/// Read once, for the reason [`gc_lifetime_log_enabled`] gives: the gate opens
/// `free_arena`, which runs at the end of every completed sweep, and upstream's
/// whole-arena-empty branch (minimarkpage.py:404-411) touches no environment
/// at all.
#[inline]
pub fn gc_freelist_diag_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_GC_FREELIST_DIAG").is_some());
    *ENABLED
}

/// `MAJIT_GC_NURSERY_POISON` — fill recycled nursery and old-gen memory with a
/// poison word, and audit the slice the resume-root registration is handed.
///
/// Read once, for the reason [`gc_lifetime_log_enabled`] gives, and with a
/// sharper case: the gate sits in `shadow_stack::push_resume_ref_roots`, which
/// every blackhole resume calls once per rebuilt frame and once per virtuals
/// cache, and `std::env::var_os` takes the environment lock and scans it
/// linearly whether or not the variable is set.  The arena and old-gen
/// constructors ask once each and could read the environment directly; routing
/// them through the same accessor keeps all three readers on one answer.
pub fn gc_nursery_poison_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_GC_NURSERY_POISON").is_some());
    *ENABLED
}

/// `MAJIT_GC_STRESS_TRACE_ALLOC` — minor-collect before a nursery malloc.
///
/// Read once. Presence, matching [`gc_lifetime_log_enabled`]: any value,
/// including empty, opts in. The check is one predicted-false branch on
/// the `malloc_fast` bump (`framework.py` `inline=True`); the collect
/// itself is the slow path. A list append still collects under the knob.
#[inline]
pub fn gc_stress_trace_alloc_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_GC_STRESS_TRACE_ALLOC").is_some());
    *ENABLED
}

/// Minor before a Trace-pool malloc when [`gc_stress_trace_alloc_enabled`].
///
/// `opencoder.py` `Trace._ops` and `trace_bufs` are the callers. A young
/// address copied into a Rust local and used after the append then dies
/// at that use. `malloc_fast` consults the knob itself.
#[inline]
pub fn stress_trace_pool_alloc(roots: *mut GcRef, root_count: usize) {
    if !gc_stress_trace_alloc_enabled() {
        return;
    }
    if !gc_sync::is_initialized() || gc_sync::in_gc_op() {
        return;
    }
    gc_sync::gc_op(|gc| gc.stress_trace_alloc_minor(roots, root_count));
}

/// Panic when `addr` is a nursery word that is not a live object start
/// allocated since the last minor.
///
/// Recorder stamp sites (`history.py` `*FrontendOp.value`) call this
/// before writing a `Value::Ref`. A Rust `Copy` held across
/// `record_bytes` / `stress_trace_pool_alloc` is already from-space
/// when it is stamped; no later extra-area walk can repair it.
/// [`gc_stress_trace_alloc_enabled`] is the gate, read once: unset is
/// one cached bool load.
#[inline]
#[track_caller]
pub fn assert_stamped_ref_is_live_object(addr: usize) {
    if !gc_stress_trace_alloc_enabled() {
        return;
    }
    assert_stamped_ref_is_live_object_slow(addr, std::panic::Location::caller());
}

#[cold]
#[inline(never)]
fn assert_stamped_ref_is_live_object_slow(addr: usize, location: &'static std::panic::Location) {
    if addr == 0 {
        return;
    }
    if !gc_sync::is_initialized() || !rgil::am_i_holding_the_gil() {
        return;
    }
    gc_sync::gc_query_reentrant(|gc| gc.assert_stamped_ref_is_live_object(addr, location));
}

/// `have_debug_prints_for("gc")` for the collector's own per-collection
/// sites: a collection opens no debug section of its own, so the section
/// ready bit would silence them under a `gc` prefix filter.
///
/// `have_debug_prints_for` is the `debug_print.c` helper; the call is
/// inlined the same way `OP_HAVE_DEBUG_PRINTS` inlines the ready-bit load.
#[inline(always)]
pub fn majit_log_enabled() -> bool {
    majit_ir::debug::have_debug_prints_for("gc")
}

/// `MAJIT_GC_DRAIN_CENSUS` — aggregate how much each minor collection drains.
///
/// A `PYPY_GC_NURSERY` sweep can show that run time barely moves across an 8x
/// range of nursery sizes without saying why.  Two very different causes have
/// that shape: the nursery is simply too small for the allocation rate, or
/// collections promote nearly everything they scan and therefore free nearly
/// nothing, so the next allocation collects again no matter how large the
/// nursery is.  The distinguishing number is the fraction of scanned bytes a
/// collection leaves behind, which is what this census records.
///
/// Read once — same shape as [`gc_lifetime_log_enabled`].
pub fn drain_census_enabled() -> bool {
    drain_census_dump_interval().is_some()
}

/// Set `MAJIT_GC_DRAIN_CENSUS` to a positive integer to also dump the running
/// summary every that many collections. The end-of-run summary is unreachable
/// for the runs this census is most needed on — a collection storm that has to
/// be killed rather than waited out — so those need the periodic line.
fn drain_census_dump_interval() -> Option<u64> {
    static INTERVAL: std::sync::LazyLock<Option<u64>> = std::sync::LazyLock::new(|| {
        let value = std::env::var_os("MAJIT_GC_DRAIN_CENSUS")?;
        Some(value.to_str().and_then(|v| v.parse().ok()).unwrap_or(0))
    });
    *INTERVAL
}

/// Survival-rate deciles: bucket `i` counts the collections that promoted
/// `[i*10, (i+1)*10)` percent of the bytes they found in the nursery, with a
/// fully-surviving collection landing in the last bucket.  Averages alone hide
/// the case this census exists to detect — a run split between cheap
/// collections and ones that copy the whole nursery.
const DRAIN_DECILES: usize = 10;

#[derive(Default)]
struct DrainCensus {
    minors: u64,
    nursery_size: usize,
    used_before: u64,
    promoted: u64,
    pinned: u64,
    deciles: [u64; DRAIN_DECILES],
}

static DRAIN_CENSUS: parking_lot::Mutex<Option<DrainCensus>> = parking_lot::Mutex::new(None);

/// Record one minor collection: the nursery bytes it found (`used_before`),
/// the bytes it copied out to the old generation (`promoted`), and how many
/// objects stayed behind pinned.
pub fn drain_census_record(
    used_before: usize,
    promoted: usize,
    pinned: usize,
    nursery_size: usize,
) {
    let mut guard = DRAIN_CENSUS.lock();
    let census = guard.get_or_insert_with(DrainCensus::default);
    census.minors += 1;
    census.nursery_size = nursery_size;
    census.used_before += used_before as u64;
    census.promoted += promoted as u64;
    census.pinned += pinned as u64;
    // A collection that found nothing drained nothing; count it as 0% rather
    // than dividing by zero.
    let percent = if used_before == 0 {
        0
    } else {
        // Widen before scaling: on wasm32 a nursery past ~43 MB makes
        // `used_before * 100` wrap a 32-bit `usize`, which buckets the sample
        // wrong in release and trips the overflow check in debug.
        ((promoted.min(used_before) as u64 * 100) / used_before as u64) as usize
    };
    census.deciles[(percent / 10).min(DRAIN_DECILES - 1)] += 1;
    if drain_census_dump_interval()
        .is_some_and(|interval| interval > 0 && census.minors.is_multiple_of(interval))
    {
        eprintln!("[gc-drain] {}", census.summary());
    }
}

impl DrainCensus {
    fn summary(&self) -> String {
        let minors = self.minors;
        let survived = if self.used_before == 0 {
            0.0
        } else {
            (self.promoted as f64 * 100.0) / self.used_before as f64
        };
        let deciles = self
            .deciles
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join("/");
        format!(
            "gc_drain minors={minors} nursery={} used_avg={} promoted_avg={} \
             freed_avg={} survived={survived:.1}% pinned_avg={:.2} deciles={deciles}",
            self.nursery_size,
            self.used_before / minors,
            self.promoted / minors,
            self.used_before.saturating_sub(self.promoted) / minors,
            self.pinned as f64 / minors as f64,
        )
    }
}

/// One-line summary of [`drain_census_record`], or a note that nothing was
/// recorded.
pub fn drain_census_summary() -> String {
    let guard = DRAIN_CENSUS.lock();
    let Some(census) = guard.as_ref().filter(|census| census.minors > 0) else {
        return "gc_drain minors=0".to_string();
    };
    census.summary()
}

/// Write barrier descriptor — information the JIT needs to emit write barrier checks.
///
/// From rpython/jit/backend/llsupport/gc.py WriteBarrierDescr.
#[derive(Debug, Clone)]
pub struct WriteBarrierDescr {
    /// gc.py:268: GCClass.JIT_WB_IF_FLAG
    pub jit_wb_if_flag: u64,
    /// gc.py: extract_flag_byte(jit_wb_if_flag) → byteofs
    /// Object-relative (negative = before object start, in header).
    pub jit_wb_if_flag_byteofs: i32,
    /// gc.py: extract_flag_byte(jit_wb_if_flag) → singlebyte
    pub jit_wb_if_flag_singlebyte: u8,
    /// gc.py:273: GCClass.JIT_WB_CARDS_SET (0 if no card marking)
    pub jit_wb_cards_set: u64,
    /// gc.py:274: GCClass.JIT_WB_CARD_PAGE_SHIFT
    pub jit_wb_card_page_shift: u32,
    /// gc.py: extract_flag_byte(jit_wb_cards_set) → byteofs
    pub jit_wb_cards_set_byteofs: i32,
    /// gc.py: extract_flag_byte(jit_wb_cards_set) → singlebyte
    /// Must equal -0x80 (signed) per gc.py:281 assert.
    pub jit_wb_cards_set_singlebyte: i8,
}

impl WriteBarrierDescr {
    /// gc.py extract_flag_byte: find the non-zero byte in the
    /// header-shifted flag word and return (obj_relative_byteofs, singlebyte).
    ///
    /// The returned offset is relative to the **object pointer** (not the
    /// header), matching RPython's convention where the JIT emits
    /// `load [obj + byteofs]`.  Since our header sits at `obj - GcHeader::SIZE`,
    /// the conversion is `obj_ofs = header_ofs - GcHeader::SIZE`.
    pub fn extract_flag_byte(flag: u64) -> (i32, i8) {
        let shifted = flag << crate::header::FLAG_SHIFT;
        let bytes = shifted.to_le_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b != 0 {
                let obj_ofs = i as i32 - crate::header::GcHeader::SIZE as i32;
                return (obj_ofs, b as i8);
            }
        }
        (0, 0)
    }

    /// Build a descriptor with correct byte offsets for the current
    /// header layout. gc.py:259-293 WriteBarrierDescr.__init__.
    pub fn for_current_gc() -> Self {
        let (if_flag_byteofs, if_flag_singlebyte) =
            Self::extract_flag_byte(GcFlags::GCFLAG_TRACK_YOUNG_PTRS.bits());
        let (cards_set_byteofs, cards_set_singlebyte) =
            Self::extract_flag_byte(GcFlags::GCFLAG_CARDS_SET.bits());
        // gc.py:280-281: the x86 backend relies on these two facts
        // to avoid one instruction in _write_barrier_fastpath.
        debug_assert_eq!(
            cards_set_byteofs, if_flag_byteofs,
            "CARDS_SET and TRACK_YOUNG_PTRS must be in the same byte"
        );
        debug_assert_eq!(
            cards_set_singlebyte, -0x80i8,
            "CARDS_SET must be the MSB of its byte (-0x80)"
        );
        WriteBarrierDescr {
            jit_wb_if_flag: GcFlags::GCFLAG_TRACK_YOUNG_PTRS.bits(),
            jit_wb_if_flag_byteofs: if_flag_byteofs,
            jit_wb_if_flag_singlebyte: if_flag_singlebyte as u8,
            jit_wb_cards_set: GcFlags::GCFLAG_CARDS_SET.bits(),
            jit_wb_card_page_shift: crate::collector::DEFAULT_CARD_PAGE_SHIFT,
            jit_wb_cards_set_byteofs: cards_set_byteofs,
            jit_wb_cards_set_singlebyte: cards_set_singlebyte,
        }
    }

    /// gc.py `WriteBarrierDescr.get_write_barrier_from_array_fn`.
    ///
    /// Upstream asks the CPU to cast the failing-case function; a backend
    /// without `write_barrier_from_array` answers 0. MiniMark publishes the
    /// array barrier exactly when card marking is configured, so a non-zero
    /// `jit_wb_cards_set` is that same answer.
    pub fn get_write_barrier_from_array_fn(&self) -> usize {
        if self.jit_wb_cards_set != 0 { 1 } else { 0 }
    }

    /// gc.py `WriteBarrierDescr.has_write_barrier_from_array`.
    pub fn has_write_barrier_from_array(&self) -> bool {
        self.get_write_barrier_from_array_fn() != 0
    }
}

/// GC allocator interface.
///
/// Provides allocation and collection primitives.
/// One `incminimark.py collect_step` state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GcStepTransition {
    pub old_state: u8,
    pub new_state: u8,
}

/// The `incminimark.py` collection states, under the names it declares.
///
/// A step reports one state rather than a set of bits, so these stay numbers.
/// Declaring each once mints the table
/// `every_gc_state_is_the_number_incminimark_gives_it` walks, so a state added
/// here is compared without anyone remembering to list it.
macro_rules! gc_states {
    ($($name:ident = $value:expr,)*) => {
        impl GcStepTransition {
            $(pub const $name: u8 = $value;)*
        }

        #[cfg(test)]
        const ALL_GC_STATES: &[(&str, u8)] = &[$((stringify!($name), $value),)*];
    };
}

gc_states! {
    STATE_SCANNING = 0,
    STATE_MARKING = 1,
    STATE_SWEEPING = 2,
    STATE_FINALIZING = 3,
}

impl GcStepTransition {
    /// `rgc.py is_done__states`: a major collection is done when the
    /// step ended in the starting state *and* did not begin there. A step that
    /// found no work reports `(STATE_SCANNING, STATE_SCANNING)`, which
    /// completes nothing.
    pub const fn is_done(self) -> bool {
        self.old_state != Self::STATE_SCANNING && self.new_state == Self::STATE_SCANNING
    }
}

/// `incminimark.py get_stats` values owned by the collector.
/// Every byte count includes the nursery where upstream's corresponding
/// `rgc.get_stats` selector does.  JIT assembler accounting is deliberately
/// outside this struct, as it is upstream (`jit_hooks.stats_asmmemmgr_*`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcMemoryStats {
    pub total_gc_memory: usize,
    pub total_allocated_memory: usize,
    pub peak_memory: usize,
    pub peak_allocated_memory: usize,
    pub total_arena_memory: usize,
    pub total_rawmalloced_memory: usize,
    pub peak_arena_memory: usize,
    pub peak_rawmalloced_memory: usize,
    pub nursery_size: usize,
    pub total_gc_time_ms: usize,
}

pub trait GcAllocator: Send {
    fn debug_validate_oldgen_freeblocks(&self, _site: &str) {}

    /// Allocate a fixed-size object in the nursery.
    fn alloc_nursery(&mut self, size: usize) -> GcRef;

    /// Allocate a HEADERLESS nursery object of exactly `size` bytes and return
    /// the RAW base (no GcHeader, no type word), matching the JIT headerless
    /// fast-path contract. Only a headerless-aware GC box may serve this.
    /// `MiniMarkGC` is a headered collector: its nursery walk reads a GcHeader
    /// at `obj - GcHeader::SIZE`, which a headerless object lacks, so it must
    /// never serve a headerless allocation. The default panics to turn a future
    /// misconfiguration (an interpreter that declares `headerless_structs` while
    /// leaving MiniMarkGC as the active dynasm GC) into a loud failure instead of
    /// a silently mis-based pointer + untraceable object.
    fn alloc_nursery_headerless(&mut self, _size: usize) -> GcRef {
        panic!(
            "alloc_nursery_headerless called on a headerless-unaware GcAllocator \
             (e.g. MiniMarkGC); headerless nursery allocation requires a \
             headerless-aware active dynasm GC box"
        );
    }

    /// Whether this allocator is headerless-aware: it serves
    /// [`Self::alloc_nursery_headerless`], so its nursery may hand out a raw
    /// base with no `GcHeader`. A backend reads this at compile time before
    /// it emits an inline headerless bump of `nursery_free`; a headered
    /// collector answers false and keeps every headerless allocation on its
    /// helper, whose panicking default is then the only path. An allocator
    /// whose fixed-size objects are all headerless is aware by definition.
    fn headerless_nursery(&self) -> bool {
        self.headerless_fixedsize()
    }

    /// [`Self::alloc_nursery_headerless`] for a caller that cannot survive a
    /// collection: the metainterp's jitcode tracer executes `NEW` while holding
    /// raw object pointers in its own register bank, which is not part of any
    /// root set, so a moving collection under this call would strand them. The
    /// allocator must grow instead of evacuating. The default forwards to the
    /// collecting form, which is correct for a non-moving collector.
    ///
    /// The forward is not itself a guard, so the obligation is on the
    /// implementor: a MOVING headerless-aware allocator must override this
    /// rather than inherit the forward. No current one does, and the two shapes
    /// that reach here are both safe — [`MiniMarkGC`](crate::collector::MiniMarkGC)
    /// moves but is headered, so `alloc_nursery_headerless`'s own default panics
    /// before any collection; the headerless boxes the backends install grow
    /// their pool instead of evacuating, which is exactly what this method asks
    /// for. Adding a moving headerless box without an override would reintroduce
    /// the silent stranding.
    fn alloc_nursery_headerless_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_nursery_headerless(size)
    }

    /// Allocate a fixed-size object with a known GC type id.
    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let _ = type_id;
        self.alloc_nursery(size)
    }

    /// Allocate a fixed-size typed object, collecting the nursery when full,
    /// while keeping one caller-owned GC slot live across the slow path.
    ///
    /// RPython's shadow-stack/stack-map transform exposes such a local only
    /// when the allocation reaches `collect_and_reserve`.  The default keeps
    /// the slot registered around the whole allocation for collectors without
    /// a nursery fast-path override. MiniMark overrides this so a successful
    /// bump allocation performs no dynamic root registration.
    ///
    /// # Safety
    /// `root` must point to a valid mutable `GcRef` slot for the duration of
    /// this call. `needs_write_barrier` must point to a valid mutable `bool`
    /// slot; the allocator writes whether initializing the result with `root`
    /// requires an old-to-young creation barrier.
    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        // Collectors without placement information keep the conservative
        // creation barrier. MiniMark overrides this and clears the flag for
        // the ordinary nursery result.
        unsafe { *needs_write_barrier = true };
        unsafe { self.add_root(root) };
        let result = self.alloc_nursery_typed(type_id, size);
        self.remove_root(root);
        result
    }

    /// [`Self::alloc_nursery_collecting_typed_rooted`] for a type that carries
    /// neither a finalizer nor the weakref flag.
    ///
    /// `gct_fv_gc_malloc` (`framework.py`) resolves both properties at
    /// transformation time and calls `malloc_fast` — the `inline=True` copy of
    /// `malloc_fixedsize` annotated `s_False, s_False, s_False`
    /// (`framework.py:361-382`) — whenever they are both false, which is the
    /// case for every fixed-size malloc of a plain struct. The default
    /// forwards to the livevars-span form.
    ///
    /// # Safety
    /// Same contract as [`Self::alloc_nursery_collecting_typed_rooted`], plus
    /// `type_id` must name a type with no destructor and no weakref flag.
    unsafe fn alloc_fast_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_fast_nursery_collecting_typed_roots(
                type_id,
                size,
                root,
                1,
                needs_write_barrier,
            )
        }
    }

    /// The `gc_push_roots(livevars)` form of
    /// [`Self::alloc_fast_nursery_collecting_typed_rooted`].  RPython's GC
    /// transform publishes every live GC variable, not a distinguished single
    /// one; this span is the direct ABI for allocation sites with more than one
    /// native-stack livevar.
    ///
    /// # Safety
    /// `roots` must address `root_count` contiguous mutable [`GcRef`] slots
    /// which remain valid until this call returns. `needs_write_barrier` has
    /// the same contract as the one-root form.
    unsafe fn alloc_fast_nursery_collecting_typed_roots(
        &mut self,
        type_id: u32,
        size: usize,
        roots: *mut GcRef,
        root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        // Collectors without placement information keep the conservative
        // creation barrier and the allocator's own root protocol.
        // MiniMark overrides this with `malloc_varsize`'s bump and only
        // `push_roots`s on `collect_and_reserve`.
        unsafe { *needs_write_barrier = true };
        for i in 0..root_count {
            unsafe { self.add_root(roots.add(i)) };
        }
        let result = self.alloc_nursery_typed(type_id, size);
        for i in (0..root_count).rev() {
            self.remove_root(unsafe { roots.add(i) });
        }
        result
    }

    /// Allocate a fixed-size object without triggering collection.
    ///
    /// Implementations may fall back to old-gen allocation when the nursery
    /// cannot satisfy the request.
    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef;

    /// Allocate a variable-size object (array/string).
    /// Allocate a variable-size object **as type id 0**.
    ///
    /// Type id 0 is not a sentinel: [`crate::trace::TypeRegistry::get`] indexes
    /// the table, so 0 names whichever type registered first and the object is
    /// traced with that type's layout. Prefer
    /// [`Self::alloc_varsize_typed`] wherever a descr is in reach -- which, at
    /// the JIT allocation sites, it always is.
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef;

    /// Allocate a variable-size object with a known GC type id. A collector
    /// with a type table initializes that type's registered length field before
    /// returning, matching `malloc_varsize`.
    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        // Dropping the id records the object as type 0 -- see
        // [`Self::alloc_varsize`]. That is only tolerable for an allocator
        // with no type table to record it in; a collector that has one must
        // override this.
        let _ = type_id;
        self.alloc_varsize(base_size, item_size, length)
    }

    /// Allocate a fixed-size object with type id without triggering collection.
    ///
    /// Falls back to old-gen when nursery is full. Used for jitframe
    /// allocation where input refs on the Rust stack are not yet protected
    /// by the shadow stack (Rust stack is not traced by GC, unlike RPython
    /// stack where `lltype.malloc` can safely trigger GC).
    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let _ = type_id;
        self.alloc_nursery_no_collect(size)
    }

    /// Fallible host-side form of `alloc_nursery_no_collect_typed`.
    ///
    /// MiniMark overrides this so rawmalloc failure is returned as NULL.
    /// Collectors without a distinct fallible path retain their existing
    /// allocation behavior.
    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_nursery_no_collect_typed(type_id, size)
    }

    /// [`Self::try_alloc_nursery_no_collect_typed`] for a type that carries
    /// neither a finalizer nor the weakref flag: `malloc_fast`
    /// (`framework.py:361-382`).
    ///
    /// # Safety
    /// `type_id` must name a type with no destructor and no weakref flag.
    unsafe fn try_alloc_fast_nursery_no_collect_typed(
        &mut self,
        type_id: u32,
        size: usize,
    ) -> GcRef {
        self.try_alloc_nursery_no_collect_typed(type_id, size)
    }

    /// Fallible no-collect allocation that also reports whether initializing
    /// the fresh result with a young GC reference needs a creation barrier.
    ///
    /// Collectors without placement information conservatively report
    /// `true`. MiniMark clears the flag for an ordinary nursery bump and keeps
    /// it set when the allocation spills to old-gen.
    ///
    /// # Safety
    /// `needs_write_barrier` must point to a valid mutable `bool` slot for the
    /// duration of this call.
    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { *needs_write_barrier = true };
        self.try_alloc_nursery_no_collect_typed(type_id, size)
    }

    /// [`Self::try_alloc_nursery_no_collect_typed_with_placement`] for a type
    /// that carries neither a finalizer nor the weakref flag: `malloc_fast`
    /// (`framework.py:361-382`).
    ///
    /// # Safety
    /// Same contract as
    /// [`Self::try_alloc_nursery_no_collect_typed_with_placement`], plus
    /// `type_id` must name a type with no destructor and no weakref flag.
    unsafe fn try_alloc_fast_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_nursery_no_collect_typed_with_placement(
                type_id,
                size,
                needs_write_barrier,
            )
        }
    }

    /// Allocate a variable-size object without triggering collection.
    ///
    /// Implementations may fall back to old-gen allocation when the nursery
    /// cannot satisfy the request.
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef;

    /// Allocate a stable-address object directly in old-gen.
    ///
    /// Used by host-side allocators (e.g. pyre-object `w_int_new`
    /// non-cached path) that return a raw pointer the caller holds on
    /// the Rust stack before it can be stored into a GC-tracked slot.
    /// Old-gen objects never move in MiniMark mark-sweep collection,
    /// so a subsequent minor collection cannot invalidate the pointer.
    ///
    /// Default implementation routes to
    /// `alloc_nursery_no_collect_typed` so backends without a
    /// distinct old-gen still compile; backends with a real old-gen
    /// override to force placement there.
    ///
    /// Non-collecting, unlike its structural counterpart
    /// `external_malloc` (incminimark.py), which tests
    /// `threshold_reached(raw_malloc_usage(totalsize))` before allocating and
    /// drives `minor_collection_with_major_progress` when it holds. That check
    /// cannot be made here: an RPython caller's locals are shadow-stack roots,
    /// so upstream may collect mid-allocation, while the callers this entry
    /// exists for are holding the raw pointer on the Rust stack precisely
    /// because it is not a root. Old-gen growth is instead answered by the
    /// interpreter safepoint's `threshold_reached` poll, which runs where the
    /// root set is known (`pyre-object`'s `gc_interp::safepoint`).
    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_nursery_no_collect_typed(type_id, size)
    }

    /// `external_malloc(typeid, 0, alloc_young=True)` (incminimark.py) for a
    /// caller that needs a stable address: the block never moves, and it is
    /// YOUNG, so the next minor collection frees it unless a root or a traced
    /// edge reaches it, and a young pointer stored into it needs no barrier.
    ///
    /// MiniMarkGC matches `external_malloc`: it tests
    /// `threshold_reached(raw_malloc_usage(totalsize))` and, when that holds,
    /// runs `minor_collection_with_major_progress` before the birth. Callers
    /// must present a complete root set across the call (`FrameBox::new`'s
    /// `push_roots` bracket; a residual CALL's gcmap). A collector without a
    /// young non-moving arm answers with [`Self::alloc_oldgen_typed`].
    fn alloc_young_nonmoving_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_oldgen_typed(type_id, size)
    }

    /// [`Self::alloc_young_nonmoving_typed`] without the collection in front
    /// of the birth, on [`Self::alloc_oldgen_typed`]'s reasoning: the caller
    /// holds the words it is about to store on the Rust stack rather than in
    /// a root, so the threshold is answered by the deferred major request and
    /// the interpreter safepoint's poll instead of here. A collector without
    /// a young non-moving arm answers with [`Self::alloc_oldgen_typed`].
    fn alloc_young_nonmoving_typed_no_collect(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_oldgen_typed(type_id, size)
    }

    /// incminimark.py:1569: jit_remember_young_pointer(obj)
    /// Perform a write barrier check on `obj`.
    /// Must be called before storing a GC reference into `obj`.
    fn write_barrier(&mut self, obj: GcRef);

    /// Write barrier for an object the caller has already proved belongs to
    /// this collector.  Fresh results from `alloc_oldgen_typed` and the
    /// collector's no-collect allocator satisfy this contract.  Collectors
    /// may skip their defensive hybrid-heap membership query; the default
    /// preserves the safe entry point for backends without that distinction.
    fn write_barrier_managed(&mut self, obj: GcRef) {
        self.write_barrier(obj);
    }

    /// incminimark.py `write_barrier_from_array(addr_array, index)`: the
    /// host `setarrayitem_gc` barrier. When card marking is configured and
    /// `TRACK_YOUNG_PTRS` is set, mark the card for `index`; otherwise the
    /// generic remembered-set barrier.
    fn write_barrier_from_array(&mut self, obj: GcRef, index: usize) {
        let _ = index;
        self.write_barrier(obj);
    }

    /// incminimark.py `writebarrier_before_move`: generalize `obj`'s cards
    /// before its items are permuted in place.  A move stores no new
    /// reference, so [`Self::write_barrier`] does not answer for it; what
    /// changes is which card page holds each pointer, and a minor reaches a
    /// carded array only through its dirty pages.  The default is the no-op a
    /// collector without card marking wants.
    fn writebarrier_before_move(&mut self, _obj: GcRef) {}

    /// incminimark.py `writebarrier_before_copy`: same effect as a barrier on
    /// each copied element, except it may reset flags a bit too eagerly.
    ///
    /// The default is the generic answer every collector can give: a
    /// write-barrier on `dest` and `true` (the copy is handled).
    fn writebarrier_before_copy(
        &mut self,
        source: GcRef,
        dest: GcRef,
        source_start: usize,
        dest_start: usize,
        length: usize,
    ) -> bool {
        let _ = (source, source_start, dest_start, length);
        self.write_barrier(dest);
        true
    }

    /// `llop.shrink_array(Bool, p, smallerlength)`, incminimark.py.
    ///
    /// Record in place that a varsize object is shorter than it was.  The
    /// default declines, which is a complete answer: `rgc.ll_shrink_array`
    /// allocates a smaller object and copies whenever the GC says no
    /// (`rgc.py`), so a collector that cannot resize anything is
    /// served by that path alone.
    fn shrink_array(&mut self, _addr: usize, _smaller_length: usize) -> bool {
        false
    }

    /// incminimark.py jit_remember_young_pointer_from_array:
    /// Called by JIT when TRACK_YOUNG_PTRS set but CARDS_SET not.
    /// Tries to set CARDS_SET if HAS_CARDS; else generic barrier.
    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef);

    /// incminimark.py remember_young_pointer_from_array2:
    /// Full card-marking barrier with index. Called when marking a
    /// specific card after CARDS_SET is already established.
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    );

    /// Trigger a minor (nursery) collection.
    fn collect_nursery(&mut self);

    /// Trigger a full collection.
    fn collect_full(&mut self);

    /// `incminimark.py collect(gen=2)`: "Do a minor (gen=0), start a major
    /// (gen=1), or do a full major (gen>=2) collection."
    ///
    /// The generation argument is the same one [`get_objects`](Self::get_objects)
    /// takes, and the two agree on what a generation is: 0 is the nursery and
    /// anything above it is what a minor collection has already promoted out of
    /// it. This is what app-level `gc.collect(n)` reaches, so the dispatch has
    /// to happen behind whichever hook the active backend installed rather than
    /// on `gc_sync`'s singleton — the wasm backend keeps its GC somewhere that
    /// singleton does not reach.
    ///
    /// A collector with no state machine has no generation to select, so the
    /// default answers every one of them with the whole collection.
    fn collect_generation(&mut self, _generation: i64) {
        self.collect_full();
    }

    /// `incminimark.py collect_step`: perform one minor collection
    /// and exactly one major-collection state transition, independently of
    /// the automatic-collection enabled flag.
    ///
    /// The default is `rgc.py`'s non-incremental answer: a collector
    /// with no state machine does the whole collection at once and reports
    /// `_encode_states(1, 0)`, a transition [`GcStepTransition::is_done`]
    /// accepts. Reporting the starting state on both sides would instead say
    /// the collection is still in progress, so a caller stepping until done
    /// would run a full collection on every iteration and never stop.
    fn collect_step(&mut self) -> GcStepTransition {
        self.collect_full();
        GcStepTransition {
            old_state: GcStepTransition::STATE_MARKING,
            new_state: GcStepTransition::STATE_SCANNING,
        }
    }

    /// `rpython/rlib/rgc.py do_get_objects`: walk every object reachable
    /// from the GC roots and visit the `rclass.OBJECT` instances selected by
    /// the generation argument. The visitor runs before the collector releases
    /// its inspection pause, so callers can root every returned object without
    /// exposing unrooted raw addresses to another collection. Collectors
    /// without an inspector visit no objects.
    fn get_objects(&mut self, _generation: i8, _visitor: &mut dyn FnMut(GcRef)) {}

    /// `pypy/module/gc/referents.py _list_w_obj_referents`: visit the
    /// app-level objects `obj` refers to directly, expanding the
    /// interpreter-internal structs in between. Same rooting contract as
    /// [`GcAllocator::get_objects`]. Collectors without an inspector visit no
    /// objects.
    fn get_referents(&mut self, _obj: GcRef, _visitor: &mut dyn FnMut(GcRef)) {}

    /// Whether the object graph reachable from `roots` contains an object
    /// whose app-level finalizer has not run yet.  A collector without graph
    /// metadata must answer conservatively: callers use `false` to skip a
    /// collection whose only purpose is prompt finalization.
    fn subgraph_has_pending_finalizer(&mut self, _roots: &[GcRef]) -> bool {
        true
    }

    /// Whether the collector traverses references out of `obj` — what
    /// `gc.is_tracked` reports. Collectors without an inspector track nothing.
    fn is_tracked(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// `rgc.get_gcflag_extra` / `toggle_gcflag_extra` /
    /// `get_gcflag_dummy`. Collectors without RPython header flags report the
    /// two queries false and reject mutation.
    fn get_gcflag_extra(&mut self, _obj: GcRef) -> bool {
        false
    }

    fn toggle_gcflag_extra(&mut self, _obj: GcRef) {
        panic!("toggle_gcflag_extra is unsupported by this collector")
    }

    fn get_gcflag_dummy(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// `inspector.py:get_rpy_memory_usage`: size of the object's translated
    /// payload, excluding the GC header and anything reachable from it.
    /// `None` is the untranslated equivalent of inspector.py returning `-1`
    /// for a collector that does not implement the operation.
    fn get_rpy_memory_usage(&mut self, _obj: GcRef) -> Option<usize> {
        None
    }

    /// `inspector.py:get_rpy_type_index`: positive member index in the
    /// translated type-info group.  Index zero is deliberately unused by
    /// RPython's `TypeLayoutBuilder.make_type_info_group`.
    fn get_rpy_type_index(&mut self, _obj: GcRef) -> Option<usize> {
        None
    }

    /// `inspector.py:get_rpy_roots`: visit the collector's raw roots without
    /// expanding interpreter-internal objects.  `false` means the collector
    /// does not implement the inspector operation.
    fn get_rpy_roots(&mut self, _visitor: &mut dyn FnMut(GcRef)) -> bool {
        false
    }

    /// `inspector.py:get_rpy_referents`: visit the raw pointers traced from
    /// one object, without the app-level expansion performed by
    /// [`GcAllocator::get_referents`].
    fn get_rpy_referents(&mut self, _obj: GcRef, _visitor: &mut dyn FnMut(GcRef)) -> bool {
        false
    }

    /// `inspector.py dump_rpy_heap`. The fd receives native signed
    /// machine words; `Ok(false)` denotes a collector without an inspector.
    fn dump_rpy_heap(&mut self, _fd: i32) -> Result<bool, i32> {
        Ok(false)
    }

    /// Translation-time type-info metadata returned by `get_typeids_z` and
    /// `get_typeids_list`. `None` denotes a collector without this operation.
    fn get_typeids_text(&self) -> Option<Vec<u8>> {
        None
    }

    fn get_typeids_list(&self) -> Option<Vec<usize>> {
        None
    }

    /// `incminimark.raw_malloc_memory_pressure(size, adr)`. When `object` is
    /// non-null, the translated `special_memory_pressure` field on that
    /// object is set to `size` before the collection threshold is adjusted.
    fn add_memory_pressure(&mut self, _size: isize, _object: GcRef) {}

    /// `incminimark.external_malloc`: when `threshold_reached(totalsize)`,
    /// run `minor_collection_with_major_progress(force_enabled=False)` with
    /// extrasize `totalsize + nursery_size/2`. Automatic major progress
    /// stays gated on `enabled`; explicit `collect(0)` is the forcing path.
    fn maybe_collect_for_external_malloc(&mut self, _totalsize: usize) -> bool {
        false
    }

    /// `inspector.count_memory_pressure`: sum the pressure fields on all
    /// root-reachable objects. Collectors without translated type metadata
    /// report zero.
    fn total_memory_pressure(&mut self) -> isize {
        0
    }

    /// Whether a raw inspector node is an app-level `W_Root` rather than an
    /// interpreter-internal GC struct that must be wrapped in `GcRef`.
    fn is_app_level_object(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// Trigger a non-moving old-gen-only major collection (sweep dead old-gen
    /// objects without moving the nursery). The default no-ops so a backend
    /// with no incremental old-gen lacks no method; `MiniMarkGC` overrides it.
    fn collect_oldgen_nonmoving(&mut self) {}

    /// Toggle automatic major-collection progress. Backends without
    /// incremental major collection ignore it.
    fn enable(&mut self) {}

    fn disable(&mut self) {}

    fn isenabled(&self) -> bool {
        true
    }

    /// rgc.py `FinalizerQueue.register_finalizer` /
    /// framework.py `gc_fq_register`: register `obj` with one translated
    /// finalizer handler.  Backends without finalizer queues ignore it.
    fn register_finalizer(&mut self, _fq_index: usize, _obj: GcRef) {}

    /// The gctransformer's `finalizer_queue_indexes` / `finalizer_handlers`
    /// for this collector's `gcdata`. `None` when this allocator has no
    /// FinalizerQueue table.
    fn finalizer_queue_table(
        &self,
    ) -> Option<std::sync::Arc<crate::collector::FinalizerQueueTable>> {
        None
    }

    /// incminimark.py `ignore_finalizer` / framework.py
    /// `gct_gc_ignore_finalizer`: mark `obj` so the finalizer queues skip it.
    /// An optimization hint — a backend with no such flag ignores it, and the
    /// finalizer simply runs.
    fn ignore_finalizer(&mut self, _obj: GcRef) {}

    /// minimark.py `id_or_identityhash(gcobj)`.
    /// Return a stable address for the object that does not change
    /// across GC moves.  For nursery objects, allocates a shadow in
    /// old-gen and returns its address.  For old-gen objects, returns
    /// the object's own address.
    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        obj_addr
    }

    /// `id_or_identityhash` without allocating a shadow.
    ///
    /// A root walk already holds the collector. `MiniMarkGC` reads an
    /// existing shadow or forwarding pointer. A stub returns `obj_addr`.
    fn id_or_identityhash_reentrant(&self, obj_addr: usize) -> usize {
        obj_addr
    }

    /// `gc.py self.write_barrier_descr = WriteBarrierDescr(self)`:
    /// the descriptor for the write barrier check. Defaulting to `None` is
    /// `gc.py GcLLDescr_boehm.write_barrier_descr = None` — a collector
    /// that needs no write barrier. Upstream reads the attribute directly and
    /// has no `get_write_barrier_descr` accessor.
    fn get_write_barrier_descr(&self) -> Option<WriteBarrierDescr> {
        None
    }

    /// Register a stack/root slot that contains a `GcRef`.
    ///
    /// The pointer must remain valid until removed. Backends use this to
    /// expose shadow-root buffers around collecting helper calls.
    ///
    /// # Safety
    /// The caller must ensure the slot remains valid for the duration of the
    /// registration.
    unsafe fn add_root(&mut self, _root: *mut GcRef) {}

    /// Remove a previously-registered root slot.
    fn remove_root(&mut self, _root: *mut GcRef) {}

    /// Whether `addr` lies inside this GC's managed heap (nursery or
    /// old-gen). Used by host-side allocators to discriminate
    /// GC-allocated blocks from `std::alloc`-backed ones during the
    /// L1/L2 stepping-stone window — `dealloc_items_block` must
    /// early-return for GC-managed pointers (the GC sweeps them) and
    /// fall through to `std::alloc::dealloc` for non-managed ones.
    /// Default `false` matches stub allocators (no managed heap).
    fn is_managed_heap_object(&self, _addr: usize) -> bool {
        false
    }

    /// Whether `addr` is a live object in the moving nursery.  Custom trace
    /// hooks use this to distinguish a nursery child whose own type walker
    /// will scan its items from an old-gen child whose owner must scan an
    /// off-barrier payload during a minor collection.
    fn is_nursery_object(&self, _addr: usize) -> bool {
        false
    }

    /// The nursery's `[start, end)`, when it is a fixed range for the life of
    /// this allocator.  `None` (the default) keeps callers on the full
    /// `is_nursery_object` query.
    fn nursery_bounds(&self) -> Option<(usize, usize)> {
        None
    }

    /// `gc/base.py is_valid_gc_object`'s tagged-immediate setting
    /// (`translationoption.py taggedpointers`, default off).
    fn taggedpointers(&self) -> bool {
        false
    }

    /// Current nursery free pointer.
    fn nursery_free(&self) -> *mut u8;

    /// gc.py get_nursery_free_addr parity.
    /// Address of the mutable nursery_free field that JIT code updates.
    fn nursery_free_addr(&self) -> usize;

    /// Nursery top (end) pointer.
    fn nursery_top(&self) -> *const u8;

    /// gc.py get_nursery_top_addr parity.
    /// Address of the published nursery_top slot that JIT code reads.
    fn nursery_top_addr(&self) -> usize;

    /// Addresses of `nursery_free` and `nursery_top` plus `nonlarge_max + 1`
    /// (`large_object_threshold`) for an inlined `malloc_fast` bump.
    ///
    /// `framework.py` `getfn(malloc_fast, …, inline=True)` copies
    /// `IncrementalMiniMarkGC.malloc_fixedsize` so the call site itself
    /// loads `nursery_free`, adds the (constant-folded) size, compares
    /// `nursery_top`, and only then calls `collect_and_reserve`. The
    /// field addresses are what `gc.py` `get_nursery_free_addr` /
    /// `get_nursery_top_addr` publish for compiled code; the interpreter
    /// malloc uses the same pair. `None` keeps the public trampoline on
    /// `gc_op`.
    fn malloc_fast_bump_addrs(&self) -> Option<(usize, usize, usize)> {
        None
    }

    /// Address of the head of a singly-linked recycle list an inline
    /// allocation may take a cell from before it bumps, or 0 when this
    /// allocator keeps none.
    ///
    /// A cell on the list is threaded through its second word — the same word
    /// a two-word headerless allocation publishes as its link — so taking the
    /// head is a load of the head, a load of that cell's link, and a store
    /// back. The default keeps inline allocation a pure bump, which is what an
    /// allocator that recycles nothing wants; one that DOES recycle has to
    /// report the list here, because a bump-only fast path goes dead as soon
    /// as the current chunk fills and the allocator starts living off the list.
    fn nursery_recycle_list_addr(&self) -> usize {
        0
    }

    /// Address of a two-word `[base, width]` pair naming the byte range whose
    /// cells the recycle list accepts, or 0 when this allocator publishes
    /// none.
    ///
    /// Publishing it alongside [`Self::nursery_recycle_list_addr`] declares
    /// that releasing a cell whose address satisfies the unsigned test
    /// `addr - base < width` is exactly `cell.link = *head; *head = cell`,
    /// with the link word at offset 8 — the same two-word headerless cell the
    /// inline nursery bump initialises. A cell that fails the test has to
    /// reach the ordinary helper, which is also where the whole release goes
    /// when either address is 0.
    fn nursery_recycle_window_addr(&self) -> usize {
        0
    }

    /// Maximum size for nursery allocation (larger objects go to old gen directly).
    fn max_nursery_object_size(&self) -> usize;

    /// incminimark.py: card_page_indices → JIT_WB_CARD_PAGE_SHIFT.
    /// Log2 of the card page size. 0 if card marking is disabled.
    fn card_page_shift(&self) -> u32 {
        0
    }

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Adds the object directly to the remembered set. The JIT has already
    /// performed the inline flag test (COND_CALL_GC_WB) and determined
    /// that the barrier is needed.
    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.write_barrier(obj);
    }

    /// `gcreftracer.py` `llop.gc_writebarrier(tr)`: a newly written table
    /// enters this collector's remembered set for one minor.
    fn remember_gc_table(&mut self, _table: &std::sync::Arc<crate::GcTable>) {}

    /// Whether the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT emits COND_CALL_GC_WB (inline flag test +
    /// conditional call) instead of a full barrier call.
    fn can_optimize_cond_call(&self) -> bool {
        false
    }

    /// `GCBase.can_optimize_clean_setarrayitems`: whether the GC transform may
    /// omit barriers for stores that only clear a freshly allocated array.
    /// Collectors with card marking must return false because those stores can
    /// otherwise leave the card bitmap clean while overwriting young edges.
    fn can_optimize_clean_setarrayitems(&self) -> bool {
        false
    }

    /// Perform one incremental GC step at a JIT safepoint.
    /// Returns true if any GC work was done.
    fn gc_step(&mut self) -> bool {
        false
    }

    /// Pin a nursery object so it won't move during minor collection.
    /// Returns true if pinning succeeded.
    fn pin(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// Unpin a previously pinned object.
    fn unpin(&mut self, _obj: GcRef) {}

    /// Check if an object is pinned.
    fn is_pinned(&self, _obj: GcRef) -> bool {
        false
    }

    /// Register a GC type descriptor and return its type id.
    ///
    /// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
    fn register_type(&mut self, _info: TypeInfo) -> u32 {
        0
    }

    /// Number of registered GC types.
    fn type_count(&self) -> usize {
        0
    }

    /// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)`.
    /// `oldgen_total_bytes` is `get_total_memory_used` (promoted + raw/large
    /// old-gen objects, NOT the nursery); `nursery_used_bytes` is the current
    /// nursery bump-pointer fill. Used to split GC-retained memory from
    /// host-heap allocations when diagnosing growth. Default `(0, 0)` for stub
    /// allocators with no byte accounting.
    fn heap_byte_stats(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Full incminimark memory/time statistics. Stub allocators report zeros.
    fn gc_memory_stats(&self) -> GcMemoryStats {
        GcMemoryStats::default()
    }

    /// incminimark.py `threshold_reached(0)`: whether the memory the
    /// collector is responsible for has caught up to the threshold it set for
    /// the next major collection. Everything that shapes that threshold — the
    /// growth ratio, `growth_rate_max`, `max_delta`, `min_heap_size`,
    /// `max_heap_size` — lives behind this answer, so a caller that cannot
    /// drive collection from the allocator can ask here instead of modelling
    /// heap growth itself. Default `false` for stub allocators with no
    /// threshold accounting.
    fn major_threshold_reached(&self) -> bool {
        false
    }

    /// Diagnostic only: `(minor_collections, major_collections)` run so far.
    /// Used to attribute run time to collection cadence (e.g. old-gen churn
    /// driving repeated majors). Default `(0, 0)` for stub allocators.
    fn collection_counts(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Minor collections run since the last major finished.
    ///
    /// Not derivable from [`collection_counts`](Self::collection_counts): that
    /// pair is cumulative, and the caller cannot subtract a snapshot it was
    /// never handed. `gc.get_count`'s second element is this number. Default
    /// `0` for stub allocators, which run no collections to count.
    fn minor_collections_since_major(&self) -> usize {
        0
    }

    /// Whether a JIT inline nursery bump of `type_id` is equivalent to
    /// `alloc_with_type`'s fast path: the type registers no destructor and is
    /// not a weakref (either would need a side-list push at allocation, i.e.
    /// the slow path). Mirrors rewrite.py's malloc fast-path eligibility
    /// (types with finalizers/weakrefs keep the call). Default `false` so
    /// stub allocators keep the helper path.
    fn type_alloc_is_plain(&self, _type_id: u32) -> bool {
        false
    }

    /// Look up the fixed-object size for a registered GC type.
    ///
    /// RPython parity: this matches `cpu.bh_new(typedescr)` reading
    /// `typedescr.size` (llmodel.py / descr.py).  Default `None` keeps
    /// stub allocators (e.g. wasm/dynasm) from claiming knowledge.
    fn type_size(&self, _type_id: u32) -> Option<usize> {
        None
    }

    /// Return the registered variable-size layout for `obj`.
    /// Fixed-size, foreign, null, and unknown objects return `None`.
    fn varsize_layout(&self, _obj: GcRef) -> Option<GcVarSizeLayout> {
        None
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Maps a vtable pointer to its registered GC type id. RPython
    /// computes this arithmetically from the GC type_info_group base
    /// (gc.py:584-589); pyre's GC keeps an explicit vtable→type_id table
    /// because pyre frontends register vtables independently from the
    /// translator pipeline.
    ///
    /// Default `None` matches a GC layer with no installed mapping
    /// (e.g. dynasm/wasm stubs). The cmp_guard_class fallback panics
    /// instead of silently producing wrong code.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, _classptr: usize) -> Option<u32> {
        None
    }

    /// Register a vtable pointer as the canonical class for a type id.
    /// Frontends call this once per type after `register_type`, mirroring
    /// how RPython's translator emits the vtable→typeid pair into the
    /// GC type_info_group.
    fn register_vtable_for_type(&mut self, _vtable: usize, _type_id: u32) {}

    /// `gctypelayout.encode_type_shapes_now` parity
    /// (gctypelayout.py): closes the type-registration phase.
    /// After freeze, `register_type` is forbidden, the
    /// `type_info_group` base address is stable, and every
    /// `is_object` type's `subclassrange_{min,max}` reflects the
    /// preorder of its inheritance chain (`assign_inheritance_ids`,
    /// rtyper/normalizecalls.py).
    ///
    /// Backends call this from `set_gc_allocator` so the embedded
    /// codegen-time pointers and bounds are immutable thereafter.
    /// Default no-op for stub allocators with no type table.
    fn freeze_types(&mut self) {}

    /// Assign `subclassrange_{min,max}` now
    /// (`normalizecalls.py assign_inheritance_ids`) without publishing
    /// `frozen_layout_table`. Idempotent. `freeze_types`
    /// (`gctypelayout.py encode_type_shapes_now`) calls this before it
    /// closes the registry.
    ///
    /// Default no-op: an allocator without a type registry has no
    /// inheritance ids to assign.
    fn assign_inheritance_ids_now(&mut self) {}

    /// Whether `frozen_layout_table` has been published, so
    /// `get_translated_info_for_typeinfo` (`llsupport/gc.py`) can embed
    /// its base address.
    ///
    /// Default `true`: an allocator without a type registry has no table
    /// to publish and is already closed.
    fn types_frozen(&self) -> bool {
        true
    }

    /// Whether this allocator keeps a type table at all
    /// (`gctypelayout.py` `TypeLayoutBuilder` — a collector translated
    /// without one has no `type_info_group` to name a shape in).
    ///
    /// A capability, deliberately, rather than a reading of `type_count()`:
    /// `register_type`'s default body returns 0 and records nothing, so an
    /// allocator with no table and one whose table is merely still empty both
    /// report zero, and the two want opposite answers from a caller deciding
    /// whether a shape id can be resolved against this allocator. The default
    /// `false` matches the same default-body allocators `freeze_types` and
    /// `supports_guard_gc_type` are written for.
    fn has_type_registry(&self) -> bool {
        false
    }

    /// Tell the descr which registered type is `JITFRAME`.
    ///
    /// `jitframe.py` declares `JITFRAME` as a static `GcStruct` and
    /// `GcLLDescription.malloc_jitframe` (gc.py) allocates it by that
    /// name; the descr never has to be told. Here the shape is registered on
    /// the collector at frontend build time (`jitframe_type_info()`), so the
    /// id it received is handed to the descr once, right after that
    /// registration and before `freeze_types`. The default drops the id: an
    /// allocator with no type table has no shape to name, and its frames are
    /// built on the host heap instead (see `jitframe_type_id`).
    fn set_jitframe_type_id(&mut self, id: u32) {
        let _ = id;
    }

    /// The `JITFRAME` type id this descr allocates frames under, or `None`
    /// for an allocator with no type table.
    ///
    /// `None` is what makes a frame a host-heap block rather than a GC
    /// object: `GcLLDescr_boehm` (gc.py) is the closest upstream shape,
    /// a non-moving descr whose frames need no root slot and no barrier. The
    /// backends read this through `majit_backend::jitframe::malloc_jitframe`
    /// and never branch on it themselves.
    fn jitframe_type_id(&self) -> Option<u32> {
        None
    }

    /// Whether this descr has a `gcrootmap`.
    ///
    /// `GcLLDescr_boehm.gcrootmap` is `None`; `GcLLDescr_framework` builds
    /// one in `_make_gcrootmap`. The backend pushes a jitframe onto a shadow
    /// stack only when `gcrootmap and gcrootmap.is_shadow_stack`
    /// (`assembler.py` `_call_header_shadowstack`). A collector that can run
    /// and has to find live jitframes is exactly "the installed descr has a
    /// gcrootmap". Default `true` keeps every existing allocator on that path.
    fn has_gcrootmap(&self) -> bool {
        true
    }

    /// llsupport/gc.py:162 / gc.py:318 `supports_guard_gc_type` flag.
    /// `GcLLDescr_boehm` sets it to `False`; `GcLLDescr_framework` sets
    /// it to `True`. Relayed to `cpu.supports_guard_gc_type` via
    /// `llmodel.py:63`. Gates the backend's `genop_guard_guard_gc_type`,
    /// `genop_guard_guard_is_object`, and `genop_guard_guard_subclass`
    /// (x86/assembler.py, 1925, 1946 `assert`) and
    /// `ConstPtrInfo.get_known_class(cpu)` at info.py. The default
    /// `false` matches `AbstractCPU.supports_guard_gc_type` in
    /// `rpython/jit/backend/model.py` and keeps backends without an
    /// installed TYPE_INFO table from emitting the guards.
    fn supports_guard_gc_type(&self) -> bool {
        false
    }

    /// Fixed-size nursery objects are raw bumps: vtable at offset 0, no
    /// `GcHeader`. `rewrite.py gen_malloc_nursery` then emits
    /// `CALL_MALLOC_NURSERY` as `CallMallocNurseryHeaderless`. Default
    /// false is `GcLLDescr_framework`, whose objects carry a header.
    fn headerless_fixedsize(&self) -> bool {
        false
    }

    /// llsupport/gc.py `check_is_object` parity. Reads the
    /// typeid for `gcref` (gc.py `get_actual_typeid`) and
    /// returns whether that type has `rclass.OBJECT` layout — i.e.
    /// whether `T_IS_RPYTHON_INSTANCE` is set in its infobits (gc.py:
    /// 631-642 walks the TYPE_INFO table to test that bit).
    ///
    /// Exposed on `cpu.check_is_object(gcptr)` via llmodel.py,
    /// which asserts `supports_guard_gc_type` before delegating. The
    /// optimizer consults this through info.py:766 inside
    /// `ConstPtrInfo.get_known_class(cpu)` to decide whether reading
    /// offset 0 of a constant gcref is safe.
    ///
    /// Returns `false` for null pointers and for backends without a
    /// type registry (matching `GcLLDescr_boehm`, which does not
    /// define `check_is_object`).
    fn check_is_object(&self, _gcref: GcRef) -> bool {
        false
    }

    /// gc/base.py `is_valid_gc_object` tagged-immediate test:
    /// `config.taggedpointers && (addr & 1 == 1)`. Backends with no
    /// tagged-immediate support (the default) return `false`.
    fn is_tagged_immediate(&self, _addr: usize) -> bool {
        false
    }

    /// `rpython/rlib/rgc.py` `can_move(p)` — whether the GC object
    /// `gcref` sits at an address that may still move. "With non-moving
    /// GCs, it is always False; with moving GCs it can be True for some
    /// time, then False once the object is sure not to move." The default
    /// is `false`, matching a non-moving GC (and the no-GC case).
    fn can_move(&self, _gcref: GcRef) -> bool {
        false
    }

    /// llsupport/gc.py `get_translated_info_for_typeinfo`.
    /// Returns `(type_info_group_base, shift_by, sizeof_ti)`:
    ///  * `type_info_group_base` — base address of the `TYPE_INFO` table
    ///    (`llop.gc_get_type_info_group`).
    ///  * `shift_by` — `2` on 32-bit, `0` on 64-bit (gc.py).
    ///  * `sizeof_ti` — `rffi.sizeof(GCData.TYPE_INFO)`.
    ///    Called by `genop_guard_guard_is_object` (x86/assembler.py)
    ///    and `genop_guard_guard_subclass` (x86/assembler.py).
    ///
    /// Default panics to match RPython: `GcLLDescr_boehm` does not
    /// define the method, and calling it when
    /// `supports_guard_gc_type = False` is a precondition violation.
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        panic!(
            "GcAllocator::get_translated_info_for_typeinfo called but the \
             GC has not installed a TYPE_INFO layout (see llsupport/gc.py:\
             592); callers must first check supports_guard_gc_type"
        )
    }

    /// llsupport/gc.py `get_translated_info_for_guard_is_object`.
    /// Returns `(infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE)` used by
    /// `genop_guard_guard_is_object` to locate the `infobits` byte in
    /// the `TYPE_INFO` entry and the bitmask for the
    /// `T_IS_RPYTHON_INSTANCE` flag.
    ///
    /// Default panics — same rationale as
    /// `get_translated_info_for_typeinfo`.
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        panic!(
            "GcAllocator::get_translated_info_for_guard_is_object called \
             but the GC has not installed a TYPE_INFO layout (see \
             llsupport/gc.py:619); callers must first check \
             supports_guard_gc_type"
        )
    }

    /// x86/assembler.py `cpu.subclassrange_min_offset`.
    /// Byte offset of the `subclassrange_min` field inside
    /// `rclass.CLASSTYPE`. `genop_guard_guard_subclass` uses it twice:
    /// once to read the subclassrange minimum from the object's
    /// vtable (x86/assembler.py) and once to locate the same
    /// field inside a `TYPE_INFO` entry (x86/assembler.py).
    ///
    /// Default panics — same rationale as the other TYPE_INFO helpers.
    fn subclassrange_min_offset(&self) -> usize {
        panic!(
            "GcAllocator::subclassrange_min_offset called but the GC has \
             not installed an rclass.CLASSTYPE layout (see x86/\
             assembler.py:1951); callers must first check \
             supports_guard_gc_type"
        )
    }

    /// x86/assembler.py:1971-1974 bounds lookup at codegen time:
    ///     vtable_ptr = loc_check_against_class.getint()
    ///     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
    ///     check_min = vtable_ptr.subclassrange_min
    ///     check_max = vtable_ptr.subclassrange_max
    /// Returns `(subclassrange_min, subclassrange_max)` for the class
    /// whose pointer is given, or `None` if no entry exists.
    ///
    /// Default `None` keeps backends without an installed
    /// `rclass.CLASSTYPE` layout from emitting a wrong bounds check;
    /// `genop_guard_guard_subclass` callers panic loudly when the
    /// lookup misses.
    fn subclass_range(&self, _classptr: usize) -> Option<(i64, i64)> {
        None
    }

    /// Companion to `subclass_range` keyed by typeid instead of
    /// classptr. Used by the executor's `GuardSubclass` arm after it
    /// resolves `value.typeptr` via `get_actual_typeid`
    /// (llgraph/runner.py). Default `None`.
    fn typeid_subclass_range(&self, _typeid: u32) -> Option<(i64, i64)> {
        None
    }

    /// gc.py `get_actual_typeid` parity. Reads the typeid
    /// from the GC header half-word for managed objects, or resolves
    /// the foreign object's classptr through `vtable_to_type_id` for
    /// backends that register a seam (e.g. pyre's PyObject layout).
    /// Default `None` for stubs without a type registry.
    fn get_actual_typeid(&self, _gcref: GcRef) -> Option<u32> {
        None
    }

    /// Companion to `check_is_object` keyed by typeid. Returns
    /// whether the typeid carries `T_IS_RPYTHON_INSTANCE` in its
    /// TYPE_INFO entry (gctypelayout.py). Default `None`.
    fn typeid_is_object(&self, _typeid: u32) -> Option<bool> {
        None
    }
}

static TYPE_REGISTRY_CLOSE_HOOK: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

/// Install the hook [`ensure_type_registry_closed`] runs before
/// `freeze_types` (`gctypelayout.py encode_type_shapes_now`).
///
/// A second call with the same function is a no-op. A different function
/// panics: one process-global owner closes the registry.
pub fn set_type_registry_close_hook(hook: fn()) {
    let installed = *TYPE_REGISTRY_CLOSE_HOOK.get_or_init(|| hook);
    assert!(
        std::ptr::fn_addr_eq(installed, hook),
        "set_type_registry_close_hook: a different close hook is already installed"
    );
}

/// Publish the singleton type registry if `frozen_layout_table` is not
/// already published. `get_translated_info_for_typeinfo`
/// (`llsupport/gc.py`) embeds that table's base address.
///
/// A no-op when `gc_sync` has no collector. Boxed allocators are closed by
/// `set_gc_allocator` and are not this registry.
///
/// Must not be called from inside a `gc_op` or `gc_query` closure.
/// `gc_query` is `gc_op`, and `gc_op` is not reentrant (`ReentryGuard`):
/// the close hook and the `freeze_types` below both call `gc_op`. The hook
/// therefore runs outside any `gc_sync` closure — the already-frozen check
/// uses `gc_query_reentrant`, and that shared borrow ends before the hook.
pub fn ensure_type_registry_closed() {
    if !gc_sync::is_initialized() {
        return;
    }
    debug_assert!(
        !gc_sync::in_gc_op(),
        "ensure_type_registry_closed must not run inside gc_op or gc_query"
    );
    if gc_sync::gc_query_reentrant(|gc| gc.types_frozen()) {
        return;
    }
    if let Some(hook) = TYPE_REGISTRY_CLOSE_HOOK.get().copied() {
        hook();
    }
    gc_sync::gc_op(|gc| gc.freeze_types());
}

/// Forwarding handle to the process-global GC singleton via `gc_sync`.
///
/// `&mut self` methods route through `gc_sync::gc_op`; `&self` read-only
/// queries route through `gc_sync::gc_query_reentrant` so they stay correct
/// when a collection-time extra-root walker re-enters the GC (ownership / type
/// queries) while this thread already holds the `&mut`, which a `&mut` path
/// would alias. No raw pointer at this layer.
///
/// # Thread safety
///
/// Exclusion comes from the GIL (`rgil`), which the caller holds for
/// as long as it runs pyre code, so these methods take no lock of their own.
/// Collection uses STW safepoint protocol (`gc_sync::request_stw`).
/// See gh#396 for the full free-threading GC design.
pub struct GcHandle;

// GcHandle is a zero-size marker; Send is trivially safe.
unsafe impl Send for GcHandle {}

impl GcAllocator for GcHandle {
    fn remember_gc_table(&mut self, table: &std::sync::Arc<crate::GcTable>) {
        gc_sync::gc_op(|gc| gc.remember_gc_table(table));
    }

    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery(size))
    }
    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_typed(type_id, size))
    }
    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.alloc_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
        })
    }
    unsafe fn alloc_fast_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.alloc_fast_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
        })
    }
    unsafe fn alloc_fast_nursery_collecting_typed_roots(
        &mut self,
        type_id: u32,
        size: usize,
        roots: *mut GcRef,
        root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.alloc_fast_nursery_collecting_typed_roots(
                type_id,
                size,
                roots,
                root_count,
                needs_write_barrier,
            )
        })
    }
    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_no_collect(size))
    }
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize(base_size, item_size, length))
    }
    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize_typed(type_id, base_size, item_size, length))
    }
    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_no_collect_typed(type_id, size))
    }
    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.try_alloc_nursery_no_collect_typed(type_id, size))
    }
    unsafe fn try_alloc_fast_nursery_no_collect_typed(
        &mut self,
        type_id: u32,
        size: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe { gc.try_alloc_fast_nursery_no_collect_typed(type_id, size) })
    }
    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.try_alloc_nursery_no_collect_typed_with_placement(type_id, size, needs_write_barrier)
        })
    }
    unsafe fn try_alloc_fast_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.try_alloc_fast_nursery_no_collect_typed_with_placement(
                type_id,
                size,
                needs_write_barrier,
            )
        })
    }
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize_no_collect(base_size, item_size, length))
    }
    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_oldgen_typed(type_id, size))
    }
    fn alloc_young_nonmoving_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_young_nonmoving_typed(type_id, size))
    }
    fn alloc_young_nonmoving_typed_no_collect(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_young_nonmoving_typed_no_collect(type_id, size))
    }
    fn write_barrier(&mut self, obj: GcRef) {
        // No root bracket: the barrier neither allocates nor collects, so
        // nothing inside it can move `obj`.  `do_write_barrier` /
        // `remember_young_pointer` (collector.rs) read headers, append to the
        // host-allocated `remembered_set`, and clear a tid flag — which is why
        // upstream inlines the barrier at the store (framework.py:533-537) and
        // publishes no shadow-stack root anywhere in it.  Restore the bracket
        // if the remembered set ever becomes GC-managed.
        gc_sync::gc_op(|gc| gc.write_barrier(obj))
    }
    fn write_barrier_from_array(&mut self, obj: GcRef, index: usize) {
        gc_sync::gc_op(|gc| gc.write_barrier_from_array(obj, index))
    }
    fn writebarrier_before_copy(
        &mut self,
        source: GcRef,
        dest: GcRef,
        source_start: usize,
        dest_start: usize,
        length: usize,
    ) -> bool {
        gc_sync::gc_op(|gc| {
            GcAllocator::writebarrier_before_copy(
                gc,
                source,
                dest,
                source_start,
                dest_start,
                length,
            )
        })
    }
    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.jit_remember_young_pointer_from_array(obj))
    }
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        gc_sync::gc_op_with_root(obj, |gc, obj| {
            gc.remember_young_pointer_from_array2(obj, index, card_page_shift)
        })
    }
    fn collect_nursery(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_nursery())
    }
    fn collect_full(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_full())
    }
    fn collect_generation(&mut self, generation: i64) {
        gc_sync::gc_op(|gc| gc.collect_generation(generation))
    }
    fn collect_step(&mut self) -> GcStepTransition {
        gc_sync::gc_op(|gc| gc.collect_step())
    }
    fn get_objects(&mut self, generation: i8, visitor: &mut dyn FnMut(GcRef)) {
        gc_sync::gc_op(|gc| gc.get_objects(generation, visitor))
    }
    fn get_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) {
        // Rooted like `write_barrier`, for the reason `gc_op_with_root`
        // states: a collection driven from inside the operation forwards `obj`,
        // and the reload is what keeps the raw argument from reaching
        // `do_get_referents` as a forwarding stub, which it rejects — reporting
        // a live object as having no referents at all.
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_referents(obj, visitor))
    }
    fn subgraph_has_pending_finalizer(&mut self, roots: &[GcRef]) -> bool {
        gc_sync::gc_op(|gc| gc.subgraph_has_pending_finalizer(roots))
    }
    fn is_tracked(&mut self, obj: GcRef) -> bool {
        // Same exposure as `get_referents`: a forwarded address is not a
        // managed-heap object of the type the answer is read out of.
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.is_tracked(obj))
    }
    fn get_gcflag_extra(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_gcflag_extra(obj))
    }
    fn toggle_gcflag_extra(&mut self, obj: GcRef) {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.toggle_gcflag_extra(obj))
    }
    fn get_gcflag_dummy(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_gcflag_dummy(obj))
    }
    fn get_rpy_memory_usage(&mut self, obj: GcRef) -> Option<usize> {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_rpy_memory_usage(obj))
    }
    fn get_rpy_type_index(&mut self, obj: GcRef) -> Option<usize> {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_rpy_type_index(obj))
    }
    fn get_rpy_roots(&mut self, visitor: &mut dyn FnMut(GcRef)) -> bool {
        gc_sync::gc_op(|gc| gc.get_rpy_roots(visitor))
    }
    fn get_rpy_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_rpy_referents(obj, visitor))
    }
    fn dump_rpy_heap(&mut self, fd: i32) -> Result<bool, i32> {
        gc_sync::gc_op(|gc| gc.dump_rpy_heap(fd))
    }
    fn get_typeids_text(&self) -> Option<Vec<u8>> {
        gc_sync::gc_query_reentrant(|gc| gc.get_typeids_text())
    }
    fn get_typeids_list(&self) -> Option<Vec<usize>> {
        gc_sync::gc_query_reentrant(|gc| gc.get_typeids_list())
    }
    fn add_memory_pressure(&mut self, size: isize, object: GcRef) {
        if object.is_null() {
            gc_sync::gc_op(|gc| gc.add_memory_pressure(size, object));
        } else {
            gc_sync::gc_op_with_root(object, |gc, object| gc.add_memory_pressure(size, object));
        }
    }
    fn maybe_collect_for_external_malloc(&mut self, totalsize: usize) -> bool {
        gc_sync::gc_op(|gc| gc.maybe_collect_for_external_malloc(totalsize))
    }
    fn total_memory_pressure(&mut self) -> isize {
        gc_sync::gc_op(|gc| gc.total_memory_pressure())
    }
    fn is_app_level_object(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.is_app_level_object(obj))
    }
    fn collect_oldgen_nonmoving(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_oldgen_nonmoving())
    }
    fn enable(&mut self) {
        gc_sync::gc_op(|gc| gc.enable())
    }
    fn disable(&mut self) {
        gc_sync::gc_op(|gc| gc.disable())
    }
    fn isenabled(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.isenabled())
    }
    fn register_finalizer(&mut self, fq_index: usize, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.register_finalizer(fq_index, obj))
    }
    fn ignore_finalizer(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.ignore_finalizer(obj))
    }
    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        gc_sync::gc_op(|gc| gc.id_or_identityhash(obj_addr))
    }
    fn id_or_identityhash_reentrant(&self, obj_addr: usize) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.id_or_identityhash_reentrant(obj_addr))
    }
    fn get_write_barrier_descr(&self) -> Option<WriteBarrierDescr> {
        gc_sync::gc_query_reentrant(|gc| gc.get_write_barrier_descr())
    }
    unsafe fn add_root(&mut self, root: *mut GcRef) {
        unsafe { gc_sync::gc_op_add_root(root) }
    }
    fn remove_root(&mut self, root: *mut GcRef) {
        gc_sync::gc_op(|gc| gc.remove_root(root))
    }
    fn is_managed_heap_object(&self, addr: usize) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_managed_heap_object(addr))
    }
    fn is_nursery_object(&self, addr: usize) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_nursery_object(addr))
    }
    fn nursery_free(&self) -> *mut u8 {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_free())
    }
    fn nursery_free_addr(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_free_addr())
    }
    fn nursery_top(&self) -> *const u8 {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_top())
    }
    fn nursery_top_addr(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_top_addr())
    }
    fn malloc_fast_bump_addrs(&self) -> Option<(usize, usize, usize)> {
        gc_sync::gc_query_reentrant(|gc| gc.malloc_fast_bump_addrs())
    }
    fn max_nursery_object_size(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.max_nursery_object_size())
    }
    fn card_page_shift(&self) -> u32 {
        gc_sync::gc_query_reentrant(|gc| gc.card_page_shift())
    }
    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.jit_remember_young_pointer(obj))
    }
    fn can_optimize_cond_call(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.can_optimize_cond_call())
    }

    fn can_optimize_clean_setarrayitems(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.can_optimize_clean_setarrayitems())
    }
    fn gc_step(&mut self) -> bool {
        gc_sync::gc_op(|gc| gc.gc_step())
    }
    fn pin(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op(|gc| gc.pin(obj))
    }
    fn unpin(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.unpin(obj))
    }
    fn is_pinned(&self, obj: GcRef) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_pinned(obj))
    }
    fn register_type(&mut self, info: trace::TypeInfo) -> u32 {
        gc_sync::gc_op(|gc| gc.register_type(info))
    }
    fn type_count(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.type_count())
    }
    fn heap_byte_stats(&self) -> (usize, usize) {
        gc_sync::gc_query_reentrant(|gc| gc.heap_byte_stats())
    }
    fn gc_memory_stats(&self) -> GcMemoryStats {
        gc_sync::gc_query_reentrant(|gc| gc.gc_memory_stats())
    }
    fn major_threshold_reached(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.major_threshold_reached())
    }
    fn collection_counts(&self) -> (usize, usize) {
        gc_sync::gc_query_reentrant(|gc| gc.collection_counts())
    }
    fn minor_collections_since_major(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.minor_collections_since_major())
    }
    fn type_alloc_is_plain(&self, type_id: u32) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.type_alloc_is_plain(type_id))
    }
    fn type_size(&self, type_id: u32) -> Option<usize> {
        gc_sync::gc_query_reentrant(|gc| gc.type_size(type_id))
    }
    fn varsize_layout(&self, obj: GcRef) -> Option<GcVarSizeLayout> {
        gc_sync::gc_query_reentrant(|gc| gc.varsize_layout(obj))
    }
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        gc_sync::gc_query_reentrant(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
    }
    fn register_vtable_for_type(&mut self, vtable: usize, type_id: u32) {
        gc_sync::gc_op(|gc| gc.register_vtable_for_type(vtable, type_id))
    }
    fn freeze_types(&mut self) {
        gc_sync::gc_op(|gc| gc.freeze_types())
    }
    fn assign_inheritance_ids_now(&mut self) {
        gc_sync::gc_op(|gc| gc.assign_inheritance_ids_now())
    }
    fn types_frozen(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.types_frozen())
    }
    fn has_type_registry(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.has_type_registry())
    }
    fn set_jitframe_type_id(&mut self, id: u32) {
        gc_sync::gc_op(|gc| gc.set_jitframe_type_id(id))
    }
    fn jitframe_type_id(&self) -> Option<u32> {
        gc_sync::gc_query_reentrant(|gc| gc.jitframe_type_id())
    }
    fn supports_guard_gc_type(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.supports_guard_gc_type())
    }
    fn check_is_object(&self, gcref: GcRef) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.check_is_object(gcref))
    }
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_tagged_immediate(addr))
    }
    fn can_move(&self, gcref: GcRef) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.can_move(gcref))
    }
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        gc_sync::gc_query_reentrant(|gc| gc.get_translated_info_for_typeinfo())
    }
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        gc_sync::gc_query_reentrant(|gc| gc.get_translated_info_for_guard_is_object())
    }
    fn subclassrange_min_offset(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.subclassrange_min_offset())
    }
    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        gc_sync::gc_query_reentrant(|gc| gc.subclass_range(classptr))
    }
    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        gc_sync::gc_query_reentrant(|gc| gc.typeid_subclass_range(typeid))
    }
    fn get_actual_typeid(&self, gcref: GcRef) -> Option<u32> {
        gc_sync::gc_query_reentrant(|gc| gc.get_actual_typeid(gcref))
    }
    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        gc_sync::gc_query_reentrant(|gc| gc.typeid_is_object(typeid))
    }
}

/// GC rewriter — transforms IR operations for GC integration.
///
/// Converts high-level NEW_*/SETFIELD_GC operations into:
/// - Inline nursery bump-pointer allocation (CALL_MALLOC_NURSERY)
/// - Write barrier conditional calls (COND_CALL_GC_WB)
///
/// Reference: rpython/jit/backend/llsupport/rewrite.py GcRewriterAssembler.
pub trait GcRewriter: Send {
    /// Rewrite a list of operations, inserting GC-aware code.
    ///
    /// Input and output stay `OpRc` so the rewriter can hand the same
    /// objects `rewrite.py GcRewriterAssembler.rewrite` appends to
    /// `self.newops` — no `Op` clone at the backend boundary.
    fn rewrite_for_gc(&self, ops: &[OpRc]) -> Vec<OpRc>;
    /// Rewrite with access to the constant pool.
    /// Returns (rewritten ops, gc_table gcrefs). Each `Const` box carries
    /// its own type via `Const::get_type`. The gcrefs list is the per-loop
    /// reference-constant list collected by `remove_constptr`
    /// (`GcRewriterAssembler.rewrite`'s `gcrefs_output_list`); the backend
    /// builds a `GcTable` from it and bakes its base address into the
    /// `LoadFromGcTable` loads. The caller's pool is not returned: the
    /// rewrite only reads it.
    ///
    /// The pool is borrowed. `GcRewriterAssembler.rewrite` does not copy
    /// one, and `resolve_constant` only reads it, so the caller's map is
    /// left in place.
    fn rewrite_for_gc_with_constants(
        &self,
        ops: &[OpRc],
        constants: &ConstMap<Const>,
    ) -> (Vec<OpRc>, Vec<GcRef>) {
        let _ = constants;
        (self.rewrite_for_gc(ops), Vec::new())
    }

    /// `GcRewriterAssembler.rewrite` once per segment (`assemble_bridge`).
    /// The default is `None`: a rewriter that cannot reset per-segment
    /// state must not be used to splice bridges into one trace.
    fn rewrite_for_gc_segments(
        &self,
        segments: &[&[OpRc]],
        constants: &ConstMap<Const>,
    ) -> Option<(Vec<Vec<OpRc>>, Vec<GcRef>)> {
        let _ = (segments, constants);
        None
    }
}

// ─────────────────────────────────────────────────────────────────────
// Process-global active GC allocator hooks
// ─────────────────────────────────────────────────────────────────────
//
// The metainterp / optimizer layer needs a backend-agnostic way to query
// the current CPU's GC type registry (llmodel.py check_is_object
// `cpu.check_is_object(gcptr)`). In RPython the optimizer reaches it via
// `self.optimizer.cpu`, which holds a reference to the backend-provided
// CPU object. majit has no such field; instead the live backends register
// a callback here that the metainterp can invoke without taking a
// backend dependency.

/// Process-global callback that answers `cpu.check_is_object(gcptr)` for
/// the currently active backend. Set by the backend when it installs a
/// GC runtime; cleared when the runtime is
/// unregistered.
pub type CheckIsObjectFn = fn(GcRef) -> bool;

/// Process-global callback that answers the collector's tagged-immediate
/// test (gc/base.py `is_valid_gc_object`) for the currently
/// active backend's GC: `config.taggedpointers && (addr & 1 == 1)`.
/// Lets a backend-agnostic caller decide that an odd-valued constant
/// address is an unboxed immediate rather than a heap object, without
/// reading an object header at offset 0.
pub type IsTaggedImmediateFn = fn(usize) -> bool;

/// Process-global callback that answers `gc_ll_descr.get_actual_typeid`
/// (gc.py:624-629) for the currently active backend. Returns the
/// `rffi.cast(HDRPTR, gcptr).tid` half-word for managed objects and
/// `vtable_to_type_id` for the foreign-object seam pyre uses. Paired
/// with `ACTIVE_CHECK_IS_OBJECT`; both are installed together so the
/// metainterp's guard interpretation stays consistent with the GC's
/// runtime layout assumptions.
pub type GetActualTypeidFn = fn(GcRef) -> Option<u32>;

/// Process-global callback that answers the codegen-time bounds lookup
/// `rffi.cast(rclass.CLASSTYPE, vtable_ptr).subclassrange_{min,max}`
/// from x86/assembler.py:1971-1974. Used by the executor's
/// `GuardSubclass` arm to evaluate bridges interpretively.
pub type SubclassRangeFn = fn(classptr: usize) -> Option<(i64, i64)>;

/// Process-global callback that answers the `value.typeptr
/// .subclassrange_min/max` lookup from llgraph/runner.py
/// directly by typeid. The backend installs this alongside
/// `subclass_range` so the executor can recover the object side of
/// `execute_guard_subclass` without going through a vtable pointer —
/// managed objects carry only a typeid in their GC header, and the
/// TYPE_INFO table already stores the preorder bounds in its paired
/// `CLASSTYPE` entry (gctypelayout.py add_vtable_after_typeinfo).
pub type TypeidSubclassRangeFn = fn(typeid: u32) -> Option<(i64, i64)>;

/// Process-global callback that answers `rclass.OBJECT`-layout queries
/// by typeid — "does this typeid carry `T_IS_RPYTHON_INSTANCE` in its
/// TYPE_INFO entry" (gctypelayout.py). The executor's
/// `GuardIsObject` arm calls this after resolving the object's typeid
/// via the `get_actual_typeid` seam, avoiding a second indirection
/// through `check_is_object` (which would re-resolve the typeid).
pub type TypeidIsObjectFn = fn(typeid: u32) -> Option<bool>;
/// Process-global callback that checks whether a type id indexes the active
/// collector's registered type table.
pub type IsRegisteredTypeIdFn = fn(typeid: u32) -> bool;

/// Process-global callback that answers `rgc.can_move(gcref)`
/// (rpython/rlib/rgc.py) for the currently active backend's GC. The
/// const-baking site (`x86/regalloc.py convert_to_imm`) consults
/// this before baking a `ConstPtr` immediate.
pub type CanMoveFn = fn(GcRef) -> bool;
/// Active-collector implementations of `rgc.pin`, `rgc.unpin`, and
/// `rgc._is_pinned` (`rpython/rlib/rgc.py`).
pub type PinFn = fn(GcRef) -> bool;
pub type UnpinFn = fn(GcRef);
pub type IsPinnedFn = fn(GcRef) -> bool;

global_hook!(static ACTIVE_CHECK_IS_OBJECT: CheckIsObjectFn);
global_hook!(static ACTIVE_IS_TAGGED_IMMEDIATE: IsTaggedImmediateFn);
global_hook!(static ACTIVE_GET_ACTUAL_TYPEID: GetActualTypeidFn);
global_hook!(static ACTIVE_SUBCLASS_RANGE: SubclassRangeFn);
global_hook!(static ACTIVE_TYPEID_SUBCLASS_RANGE: TypeidSubclassRangeFn);
global_hook!(static ACTIVE_TYPEID_IS_OBJECT: TypeidIsObjectFn);
global_hook!(static ACTIVE_IS_REGISTERED_TYPE_ID: IsRegisteredTypeIdFn);
global_hook!(static ACTIVE_CAN_MOVE: CanMoveFn);
global_hook!(static ACTIVE_PIN: PinFn);
global_hook!(static ACTIVE_UNPIN: UnpinFn);
global_hook!(static ACTIVE_IS_PINNED: IsPinnedFn);
static ACTIVE_SUPPORTS_GUARD_GC_TYPE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Bundle of callbacks the metainterp / executor can reach through
/// process-global cells. Mirrors the fan-out of methods RPython's optimizer
/// and blackhole reach via `self.cpu` / `self.cpu.gc_ll_descr`; majit
/// installs them together so a backend swap is a single call.
#[derive(Clone, Copy, Default)]
pub struct ActiveGcGuardHooks {
    pub check_is_object: Option<CheckIsObjectFn>,
    pub is_tagged_immediate: Option<IsTaggedImmediateFn>,
    pub get_actual_typeid: Option<GetActualTypeidFn>,
    pub subclass_range: Option<SubclassRangeFn>,
    pub typeid_subclass_range: Option<TypeidSubclassRangeFn>,
    pub typeid_is_object: Option<TypeidIsObjectFn>,
    pub is_registered_type_id: Option<IsRegisteredTypeIdFn>,
    pub can_move: Option<CanMoveFn>,
    pub pin: Option<PinFn>,
    pub unpin: Option<UnpinFn>,
    pub is_pinned: Option<IsPinnedFn>,
    pub supports_guard_gc_type: bool,
}

/// Install the active backend's GC-guard callbacks.
/// Called by backends when they enter a JIT region. Pass a default
/// `ActiveGcGuardHooks` with every field set to `None` / `false` to
/// clear. Mirrors how RPython's `cpu` field lets the optimizer and
/// executor reach `cpu.check_is_object`, `gc_ll_descr
/// .get_actual_typeid`, and the codegen-time bounds lookup; majit
/// bundles them here so a backend install is a single call.
pub fn set_active_gc_guard_hooks(hooks: ActiveGcGuardHooks) {
    ACTIVE_CHECK_IS_OBJECT.set(hooks.check_is_object);
    ACTIVE_IS_TAGGED_IMMEDIATE.set(hooks.is_tagged_immediate);
    ACTIVE_GET_ACTUAL_TYPEID.set(hooks.get_actual_typeid);
    ACTIVE_SUBCLASS_RANGE.set(hooks.subclass_range);
    ACTIVE_TYPEID_SUBCLASS_RANGE.set(hooks.typeid_subclass_range);
    ACTIVE_TYPEID_IS_OBJECT.set(hooks.typeid_is_object);
    ACTIVE_IS_REGISTERED_TYPE_ID.set(hooks.is_registered_type_id);
    ACTIVE_CAN_MOVE.set(hooks.can_move);
    ACTIVE_PIN.set(hooks.pin);
    ACTIVE_UNPIN.set(hooks.unpin);
    ACTIVE_IS_PINNED.set(hooks.is_pinned);
    ACTIVE_SUPPORTS_GUARD_GC_TYPE.store(
        hooks.supports_guard_gc_type,
        std::sync::atomic::Ordering::Release,
    );
}

/// Snapshot of the current guard-hook cells (test-only helper).
#[allow(dead_code)]
fn current_gc_guard_hooks() -> ActiveGcGuardHooks {
    ActiveGcGuardHooks {
        check_is_object: ACTIVE_CHECK_IS_OBJECT.get(),
        is_tagged_immediate: ACTIVE_IS_TAGGED_IMMEDIATE.get(),
        get_actual_typeid: ACTIVE_GET_ACTUAL_TYPEID.get(),
        subclass_range: ACTIVE_SUBCLASS_RANGE.get(),
        typeid_subclass_range: ACTIVE_TYPEID_SUBCLASS_RANGE.get(),
        typeid_is_object: ACTIVE_TYPEID_IS_OBJECT.get(),
        is_registered_type_id: ACTIVE_IS_REGISTERED_TYPE_ID.get(),
        can_move: ACTIVE_CAN_MOVE.get(),
        pin: ACTIVE_PIN.get(),
        unpin: ACTIVE_UNPIN.get(),
        is_pinned: ACTIVE_IS_PINNED.get(),
        supports_guard_gc_type: supports_guard_gc_type(),
    }
}

static GUARD_HOOKS_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Test-only: install `hooks` while holding a process-global lock, restoring
/// the previous guard-hook values when the returned guard drops. Serializes
/// hook-overriding tests so the now-global cells do not race. `pub` because
/// the sole caller lives in the `majit-metainterp` crate's tests.
pub struct GuardHooksTestGuard {
    prev: ActiveGcGuardHooks,
    _lock: parking_lot::MutexGuard<'static, ()>,
}
impl Drop for GuardHooksTestGuard {
    fn drop(&mut self) {
        set_active_gc_guard_hooks(self.prev);
    }
}
pub fn override_gc_guard_hooks_for_test(hooks: ActiveGcGuardHooks) -> GuardHooksTestGuard {
    let lock = GUARD_HOOKS_TEST_LOCK.lock();
    let prev = current_gc_guard_hooks();
    set_active_gc_guard_hooks(hooks);
    GuardHooksTestGuard { prev, _lock: lock }
}

/// Hand the collector the payload address of every jitframe that is currently
/// a live DEADFRAME, for the visitor to trace.
pub type LiveDeadFrameWalkerFn = fn(&mut dyn FnMut(usize));

global_hook!(static ACTIVE_LIVE_DEADFRAME_WALKER: LiveDeadFrameWalkerFn);

/// Backend callbacks that expose GC roots which exist only between a compiled
/// run returning and the frontend finishing with its result.
///
/// `llmodel.py` (`grab_exc_value` / `get_savedata_ref`) and
/// `llmodel.py` (`malloc_jitframe`) together say that the deadframe IS the
/// jitframe: `execute_token` allocates one JITFRAME, compiled code returns it,
/// and every `cpu.get_*_value(deadframe, i)` is a cast plus a read out of that
/// same object. Nothing is copied out and nothing else is allocated.
///
/// Upstream can hold the deadframe across arbitrary frontend work because the
/// jitframe is an ordinary GC object and `deadframe` is an ordinary local in
/// RPython code, so the translated stack map roots it for free. **A STRUCTURAL
/// ADAPTATION IS NEEDED HERE, AND THIS TABLE IS IT.** majit's jitframes for
/// the dynasm backend are libc allocations outside both the nursery and the
/// old generation; the collector reaches them only where they appear on the JF
/// shadow stack, and the compiled epilogue pops them from it before returning.
/// So between the run returning and the deadframe being dropped, the frame's
/// interior `Ref` slots are visible to nobody — and any allocation the
/// frontend makes in that window can move the objects they point at. There is
/// no conservative stack scan to fall back on.
///
/// The backend therefore publishes the set of deadframes it is holding, and
/// the collector walks it as a root source alongside the shadow stack. It is a
/// fn-pointer table rather than a registry owned by this crate for the same
/// reason [`ActiveGcGuardHooks`] is: majit-gc must not know what a deadframe
/// is, and the backend already owns the lifetime.
#[derive(Clone, Copy, Default)]
pub struct ActiveGcDeadFrameHooks {
    /// Yields each live deadframe's jitframe payload address. Interiors are
    /// walked with the tracer `shadow_stack::register_libc_jitframe_tracer`
    /// installed, the same one the shadow-stack walk uses, so a frame reached
    /// through this table and one reached on the stack are traced identically.
    pub walk_live_deadframes: Option<LiveDeadFrameWalkerFn>,
}

/// Install the active backend's deadframe-root callbacks. Pass a default
/// [`ActiveGcDeadFrameHooks`] to clear.
pub fn set_active_gc_deadframe_hooks(hooks: ActiveGcDeadFrameHooks) {
    ACTIVE_LIVE_DEADFRAME_WALKER.set(hooks.walk_live_deadframes);
}

/// Snapshot of the installed deadframe-root callbacks.
pub fn active_gc_deadframe_hooks() -> ActiveGcDeadFrameHooks {
    ActiveGcDeadFrameHooks {
        walk_live_deadframes: ACTIVE_LIVE_DEADFRAME_WALKER.get(),
    }
}

/// Visit the payload address of every jitframe currently held as a deadframe.
///
/// A no-op when no backend has installed a walker — which is also the state a
/// backend that frees its jitframe before returning leaves this in, and is
/// correct for it: such a backend holds no frame for the collector to root.
pub fn walk_active_live_deadframes(visitor: &mut dyn FnMut(usize)) {
    if let Some(f) = ACTIVE_LIVE_DEADFRAME_WALKER.get() {
        f(visitor);
    }
}

/// llmodel.py `cpu.check_is_object(gcptr)` shim. Returns whether
/// `gcref` is a `T_IS_RPYTHON_INSTANCE` (has `typeptr` at offset 0). When
/// no backend has installed a callback, returns `false`.
pub fn check_is_object(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    match ACTIVE_CHECK_IS_OBJECT.get() {
        Some(f) => f(gcref),
        None => false,
    }
}

/// gc/base.py `is_valid_gc_object` tagged-immediate test shim.
/// Delegates to the active backend's installed callback, which reads its
/// GC's `config.taggedpointers`. Returns `false` for null and when no
/// backend is installed — same absent-backend semantics as
/// `check_is_object`, so flag-off / no-GC paths are unaffected.
pub fn is_tagged_immediate(addr: usize) -> bool {
    match ACTIVE_IS_TAGGED_IMMEDIATE.get() {
        Some(f) => f(addr),
        None => false,
    }
}

/// Whether the active backend's GC has `config.taggedpointers` enabled
/// (translationoption.py:185). The installed `is_tagged_immediate` callback
/// answers `config.taggedpointers && (addr & 1 == 1)`, so probing it with an
/// odd sentinel address isolates the config flag without a live pointer.
/// Returns `false` when no backend is installed — same absent-backend
/// semantics as [`is_tagged_immediate`], so flag-off paths are unaffected.
pub fn taggedpointers_enabled() -> bool {
    is_tagged_immediate(1)
}

/// gc.py `gc_ll_descr.get_actual_typeid(gcptr)` shim.
/// Delegates to the active backend's installed callback; returns
/// `None` when no backend is installed, which mirrors
/// `llgraph/runner.py` skip semantics (the interpretive
/// guard treats an unresolved object as passing).
pub fn get_actual_typeid(gcref: GcRef) -> Option<u32> {
    if gcref.is_null() {
        return None;
    }
    match ACTIVE_GET_ACTUAL_TYPEID.get() {
        Some(f) => f(gcref),
        None => None,
    }
}

/// `rgc.can_move(gcref)` shim (rpython/rlib/rgc.py). Delegates to the
/// active backend's installed callback. Returns `false` for null pointers
/// and when no backend is installed — i.e. a non-moving / absent GC, where
/// every object address is stable (rgc.py "with non-moving GCs, it is
/// always False").
pub fn can_move(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    match ACTIVE_CAN_MOVE.get() {
        Some(f) => f(gcref),
        None => false,
    }
}

/// `rgc.pin(gcref)` shim. A missing collector has the untranslated
/// `llheap.pin` behaviour and declines the request.
pub fn pin(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    ACTIVE_PIN.get().is_some_and(|f| f(gcref))
}

/// `rgc.unpin(gcref)` shim. `rgc.unpin` is only valid after a successful
/// [`pin`], so reaching an absent collector is a caller-contract violation,
/// just as untranslated `llheap.unpin` is.
pub fn unpin(gcref: GcRef) {
    let f = ACTIVE_UNPIN
        .get()
        .expect("unpin() called without an active collector");
    f(gcref);
}

/// `rgc._is_pinned(gcref)` shim. A missing collector is non-moving and has
/// no pinned nursery objects.
pub fn is_pinned(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    ACTIVE_IS_PINNED.get().is_some_and(|f| f(gcref))
}

/// `rpython/rlib/rgc.py` `_make_sure_does_not_move`: force collections until
/// a non-null object is permanently non-moving, or decline a pinned object.
///
/// `p` is the caller's local GC reference. Its root is reread after each
/// collection and copied back on return, including when unwinding.
/// The owner-root slot supplies the live-variable save/reload that upstream's
/// framework GC transform inserts around `collect(i)`. Other live references
/// in the caller still need their own roots, as for any collecting operation.
/// The error is upstream's `NotImplementedError` for a collector that cannot
/// make the object non-moving after trying generations -1 through 6.
pub fn _make_sure_does_not_move(p: &mut GcRef) -> Result<bool, &'static str> {
    assert!(!p.is_null());
    if is_pinned(*p) {
        // A pin can be released at any time; it is not permanent immobility.
        return Ok(false);
    }
    // `BaseFrameworkGCTransformer.gct_gc__collect` saves/reloads livevars
    // across the collecting call. A native Rust unwind must reload too: the
    // collector can have moved the object before reporting a failure.
    struct ReloadRoot<'a> {
        saved: shadow_stack::OwnerRootGuard,
        local: &'a mut GcRef,
    }
    impl Drop for ReloadRoot<'_> {
        fn drop(&mut self) {
            *self.local = self.saved.get();
        }
    }
    let root = ReloadRoot {
        saved: shadow_stack::OwnerRootGuard::new(*p),
        local: p,
    };
    let mut i = -1;
    while can_move(root.saved.get()) {
        if i > 6 {
            return Err("can't make object non-movable!");
        }
        collect_generation(i);
        i += 1;
    }
    Ok(true)
}

/// x86/assembler.py:1971-1974 codegen-time bounds lookup shim used by
/// the interpretive `GuardSubclass`. Returns
/// `(subclassrange_min, subclassrange_max)` for the class whose vtable
/// pointer is given, or `None` when no backend is installed.
pub fn subclass_range(classptr: usize) -> Option<(i64, i64)> {
    ACTIVE_SUBCLASS_RANGE.get().and_then(|f| f(classptr))
}

/// Companion to `subclass_range` keyed by typeid instead of classptr.
/// Resolves `value.typeptr.subclassrange_min/max` from
/// llgraph/runner.py execute_guard_subclass when the executor only has a typeid in
/// hand (e.g. after calling `get_actual_typeid` on an object whose
/// classptr is known only to the GC). Returns `None` when no backend
/// is installed.
pub fn typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    ACTIVE_TYPEID_SUBCLASS_RANGE.get().and_then(|f| f(typeid))
}

/// Companion to `check_is_object` keyed by typeid. Called by the
/// executor's `GuardIsObject` arm after resolving the object to a
/// typeid via `get_actual_typeid`. Returns `None` when no backend is
/// installed.
pub fn typeid_is_object(typeid: u32) -> Option<bool> {
    ACTIVE_TYPEID_IS_OBJECT.get().and_then(|f| f(typeid))
}

/// Whether `typeid` is present in the active collector's registered type
/// table. Returns `false` when no backend has installed the query; callers
/// that deliberately support a no-GC fallback must check that state first.
pub fn is_registered_type_id(typeid: u32) -> bool {
    ACTIVE_IS_REGISTERED_TYPE_ID
        .get()
        .is_some_and(|f| f(typeid))
}

/// llmodel.py:63 `supports_guard_gc_type` shim. Mirrors the active
/// backend's capability flag. `false` when no backend has been installed.
pub fn supports_guard_gc_type() -> bool {
    ACTIVE_SUPPORTS_GUARD_GC_TYPE.load(std::sync::atomic::Ordering::Acquire)
}

/// Set once any thread has installed a per-thread GC box on any backend. The
/// process-global language runtime installs the backend standalone; this path
/// also serves embedders whose complete managed heap is execution-context
/// local and never shares object identity across threads.
///
/// Two live boxes at once is not a supported configuration: `ConstPtr`
/// `MAJOR_LIVE` / the intern table and `TRACE_OPS_GC_TYPE_ID` are
/// process-global indexes into one collector. Sequential install on one
/// thread (replace, then drop) is the test shape; production is one
/// `gc_sync` singleton.
///
/// Set-only, never cleared: a backend can drop one thread's box while another
/// thread still owns one, and clearing the flag would route past that live box.
///
/// RPython has no per-thread allocator — the GC transformer weaves every
/// `malloc` against the single translation-time `gcdata`
/// (`gctransform/framework.py`). A language runtime with shared managed values
/// must therefore use [`gc_sync`]; a box is valid only when the embedder's heap
/// itself has the narrower execution-context lifetime.
///
/// Only compiled when a box can exist at all, i.e. under `gc_box`; see
/// [`gc_box_installed`] for what the other build spells instead.
#[cfg(any(test, feature = "gc_box"))]
static GC_BOX_INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set once a per-thread box whose descr has a `gcrootmap` is installed.
/// Set-only, never cleared, same as [`GC_BOX_INSTALLED`]: one such box
/// anywhere in the process means a collector can run and must find frames.
#[cfg(any(test, feature = "gc_box"))]
static GC_BOX_HAS_GCROOTMAP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Record that a backend has installed a per-thread GC box.
///
/// `has_gcrootmap` is the installed allocator's [`GcAllocator::has_gcrootmap`].
/// Only a box with a `gcrootmap` sets [`GC_BOX_HAS_GCROOTMAP`]; a box without
/// one still sets [`GC_BOX_INSTALLED`] so allocation and barrier trampolines
/// route to it. Neither flag is ever cleared.
#[cfg(any(test, feature = "gc_box"))]
pub fn note_gc_box_installed(has_gcrootmap: bool) {
    GC_BOX_INSTALLED.store(true, std::sync::atomic::Ordering::Release);
    if has_gcrootmap {
        GC_BOX_HAS_GCROOTMAP.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// A build without `gc_box` answers every [`gc_box_installed`] query with a
/// constant `false`, so a box installed here would be invisible to all of them
/// and its owner would silently allocate out of the singleton instead. Refuse
/// loudly rather than let that read as a pass.
#[cfg(not(any(test, feature = "gc_box")))]
pub fn note_gc_box_installed(_has_gcrootmap: bool) {
    panic!(
        "a per-thread GC box was installed in a build without the `gc_box` feature; \
         a process-global runtime must use `install_gc_standalone`, while a test or \
         thread-confined embedder must enable `majit-gc/gc_box`"
    );
}

/// Whether any thread may own a per-thread GC box. See [`GC_BOX_INSTALLED`].
#[cfg(any(test, feature = "gc_box"))]
#[inline]
pub fn gc_box_installed() -> bool {
    GC_BOX_INSTALLED.load(std::sync::atomic::Ordering::Acquire)
}

/// No backend can hold a box in this build, so every `gc_box_installed() && …`
/// probe folds away and the allocation, write-barrier and query trampolines
/// call `gc_sync` directly — which is the whole shape RPython has.
#[cfg(not(any(test, feature = "gc_box")))]
#[inline(always)]
pub fn gc_box_installed() -> bool {
    false
}

/// Whether any installed per-thread box has a `gcrootmap`. See
/// [`GC_BOX_HAS_GCROOTMAP`].
#[cfg(any(test, feature = "gc_box"))]
#[inline]
fn gc_box_has_gcrootmap() -> bool {
    GC_BOX_HAS_GCROOTMAP.load(std::sync::atomic::Ordering::Acquire)
}

/// No box can exist in this build, so no box-owned `gcrootmap` either.
#[cfg(not(any(test, feature = "gc_box")))]
#[inline(always)]
fn gc_box_has_gcrootmap() -> bool {
    false
}

/// Whether a collector can run and therefore has to find live jitframes.
///
/// That is "the installed descr has a `gcrootmap`"
/// (`GcLLDescr_boehm.gcrootmap` is `None`; `GcLLDescr_framework` has one via
/// `_make_gcrootmap`). A per-thread box without a gcrootmap still answers
/// [`gc_box_installed`] so allocation and barrier trampolines route to it;
/// it does not make this true. `gc_sync::is_initialized` is the
/// process-global collector, which always has one.
#[inline]
pub fn collector_installed() -> bool {
    gc_box_has_gcrootmap() || gc_sync::is_initialized()
}

/// `GcLLDescr_boehm.malloc_fn_ptr`, the address `gen_malloc_fixedsize` calls.
///
/// Boehm has no inline nursery (`fielddescr_tid is None`), so `NEW` /
/// `NEW_WITH_VTABLE` are `CALL_R(malloc_fixedsize, size)` and the vtable is
/// stored at offset 0 afterwards. Unset (0) leaves the backend on its
/// existing allocator. Installing this does not install a collector:
/// [`collector_installed`] stays false.
static MALLOC_FIXEDSIZE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Publish `malloc_fixedsize`. `None` clears the hook.
pub fn set_malloc_fixedsize(func: Option<extern "C" fn(usize) -> *mut u8>) {
    MALLOC_FIXEDSIZE.store(
        func.map(|f| f as usize).unwrap_or(0),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The published `malloc_fixedsize` address, or 0 when unset.
#[inline]
pub fn malloc_fixedsize_addr() -> usize {
    MALLOC_FIXEDSIZE.load(std::sync::atomic::Ordering::Relaxed)
}

/// `GcLLDescr_boehm.malloc_array`: one call allocates through
/// `malloc_fn_ptr`, then stores the length at `lendescr`. The host
/// already returned zeroed memory, so this does not clear again.
pub extern "C" fn boehm_malloc_array(
    basesize: usize,
    num_elem: usize,
    itemsize: usize,
    ofs_length: usize,
) -> *mut u8 {
    let addr = malloc_fixedsize_addr();
    if addr == 0 {
        return core::ptr::null_mut();
    }
    let Some(var_bytes) = itemsize.checked_mul(num_elem) else {
        return core::ptr::null_mut();
    };
    let Some(total) = basesize.checked_add(var_bytes) else {
        return core::ptr::null_mut();
    };
    let func: extern "C" fn(usize) -> *mut u8 = unsafe { core::mem::transmute(addr) };
    // `gc.py` `GcLLDescr_boehm.malloc_array` calls `malloc_fn_ptr(totalsize)`.
    let res = func(total);
    if res.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        *res.add(ofs_length).cast::<usize>() = num_elem;
    }
    res
}

// ── Host-side nursery allocation hook ───────────────────────────────
//
// Separate from `ActiveGcGuardHooks` because allocation is not a
// guard-time concern. The backend installs one function pointer here
// so host-side allocators (pyre-object's `w_int_new`, `w_float_new`,
// …) can route through the real GC without taking a backend-specific
// dependency. Mirrors how RPython host code reaches `gc.malloc(TYPE)`
// through the global GC instance.
//
// When no thread owns a per-thread GC box, a backend's hook does nothing but
// forward to `gc_sync` — the same process-global singleton this crate owns. The
// `standalone_*` functions below are that forwarding, spelled once here and
// called from both sides: the backend trampolines fall through to them, and the
// entry points take them directly instead of going out through the hook. Same
// work, one fewer indirect call and stack frame per allocation. The hook is
// still consulted for its presence, so "no backend installed" keeps returning
// null exactly as before.

/// The nursery allocation a backend's `AllocNurseryTypedFn` performs once no
/// per-thread box owns the request.
///
/// Non-collecting on purpose: the caller holds a raw pointer that is not a
/// registered GC root, so a collection here would move the fresh object out
/// from under it. Falling back to old-gen on a full nursery keeps the result
/// stable across any minor collection before the caller stores it into a
/// tracked slot.
#[inline]
pub fn standalone_alloc_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    gc_sync::gc_op(|g| g.try_alloc_nursery_no_collect_typed(type_id, payload_size))
}

/// [`standalone_alloc_nursery_typed`] for a type that carries neither a
/// finalizer nor the weakref flag: `malloc_fast` (`framework.py:361-382`).
///
/// # Safety
/// `type_id` must name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn standalone_alloc_fast_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    gc_sync::gc_op(|g| unsafe { g.try_alloc_fast_nursery_no_collect_typed(type_id, payload_size) })
}

/// The rooted collecting nursery allocation a backend's
/// `AllocNurseryCollectingTypedRootedFn` performs once no per-thread box owns
/// the request.
///
/// # Safety
/// Same contract as [`alloc_nursery_collecting_typed_rooted`]: `root` and
/// `needs_write_barrier` must stay valid for the call.
#[inline]
pub unsafe fn standalone_alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    gc_sync::gc_op(|g| unsafe {
        g.alloc_nursery_collecting_typed_rooted(type_id, payload_size, root, needs_write_barrier)
    })
}

/// [`standalone_alloc_nursery_collecting_typed_rooted`] for a type that carries
/// neither a finalizer nor the weakref flag: `malloc_fast`
/// (`framework.py:361-382`).
///
/// Backend trampolines fall through here when this thread has no GC box.
/// The singleton leaf, not the published bump: a box disarms
/// [`PUBLISHED_BUMP`], and routing back through [`malloc_fast_bump`]
/// would re-enter the trampoline.
///
/// # Safety
/// Same contract as [`standalone_alloc_nursery_collecting_typed_rooted`], plus
/// `type_id` must name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn standalone_alloc_fast_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    unsafe { malloc_fast_via_gc_sync(type_id, payload_size, root, 1, needs_write_barrier) }
}

/// `malloc_fast` (`framework.py`, `inline=True`) / `malloc_varsize`
/// (`incminimark.py`): load `nursery_free`, add, compare `nursery_top`,
/// store `nursery_free`, store tid. The addresses are the same words
/// `GcLLDescr_framework.get_nursery_free_addr` /
/// `get_nursery_top_addr` publish to the JIT (`llsupport/gc.py`).
/// `collect_and_reserve`, the installation / box trampoline, the
/// stress knob, and a large-object birth are the cold tail.
///
/// # Safety
/// Same contract as [`alloc_fast_nursery_collecting_typed_roots`].
#[inline(always)]
unsafe fn malloc_fast_bump(
    type_id: u32,
    payload_size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    unsafe { *needs_write_barrier = false };
    let Some(raw) = header::GcHeader::SIZE.checked_add(payload_size) else {
        return GcRef(0);
    };
    let free_addr = PUBLISHED_BUMP
        .free_addr
        .load(std::sync::atomic::Ordering::Relaxed);
    let large = PUBLISHED_BUMP
        .large
        .load(std::sync::atomic::Ordering::Relaxed);
    if free_addr == 0 || raw >= large || gc_stress_trace_alloc_enabled() {
        return unsafe {
            malloc_fast_slow(
                type_id,
                payload_size,
                roots,
                root_count,
                needs_write_barrier,
            )
        };
    }
    let total = raw.max(header::GcHeader::MIN_NURSERY_OBJ_SIZE);
    let total = (total + header::MEMORY_ALIGNMENT - 1) & !(header::MEMORY_ALIGNMENT - 1);
    // `nursery_free` is the first field of `NurseryPtrs` (`repr(C)`),
    // so the published free address is the struct the JIT bakes.
    let ptrs = free_addr as *mut crate::nursery::NurseryPtrs;
    let result = unsafe { (*ptrs).free };
    let new_free = result.wrapping_add(total);
    if (new_free as usize) > (unsafe { (*ptrs).top } as usize) {
        return unsafe {
            malloc_fast_slow(
                type_id,
                payload_size,
                roots,
                root_count,
                needs_write_barrier,
            )
        };
    }
    unsafe {
        (*ptrs).free = new_free;
        (result as *mut u64).write(u64::from(type_id));
    }
    GcRef((result as usize) + header::GcHeader::SIZE)
}

/// Overflow, large object, unpublished bump, box, or `MAJIT_GC_STRESS_TRACE_ALLOC`.
///
/// `framework.py` `push_roots` / `pop_roots` only wrap this arm:
/// `malloc_fast` inlines the bump, and `postprocess_inlining` leaves
/// the livevars on the shadow stack solely around `collect_and_reserve`.
#[cold]
#[inline(never)]
unsafe fn malloc_fast_slow(
    type_id: u32,
    payload_size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    let _live = unsafe { crate::shadow_stack::LiveRootFrame::push(roots, root_count) };
    if gc_box_installed() {
        if let Some(f) = ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTS.get() {
            return unsafe {
                f(
                    type_id,
                    payload_size,
                    roots,
                    root_count,
                    needs_write_barrier,
                )
            };
        }
        if root_count == 1 {
            if let Some(f) = ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED.get() {
                return unsafe { f(type_id, payload_size, roots, needs_write_barrier) };
            }
        }
    }
    unsafe {
        malloc_fast_via_gc_sync(
            type_id,
            payload_size,
            roots,
            root_count,
            needs_write_barrier,
        )
    }
}

/// `collect_and_reserve` on the process singleton. No hook, no bump.
#[cold]
#[inline(never)]
unsafe fn malloc_fast_via_gc_sync(
    type_id: u32,
    payload_size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    if !gc_sync::is_initialized() {
        unsafe { *needs_write_barrier = true };
        return GcRef(0);
    }
    gc_sync::gc_op(|g| unsafe {
        if gc_stress_trace_alloc_enabled() {
            g.stress_trace_alloc_minor(roots, root_count);
        }
        g.alloc_fast_nursery_collecting_typed_roots(
            type_id,
            payload_size,
            roots,
            root_count,
            needs_write_barrier,
        )
    })
}

/// [`standalone_alloc_fast_nursery_collecting_typed_rooted`] with the complete
/// live-root span produced by an RPython-style `gc_push_roots(livevars)`.
///
/// # Safety
/// `roots` must address `root_count` contiguous mutable [`GcRef`] slots for the
/// duration of the call.
#[inline]
pub unsafe fn standalone_alloc_fast_nursery_collecting_typed_roots(
    type_id: u32,
    payload_size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    unsafe {
        malloc_fast_via_gc_sync(
            type_id,
            payload_size,
            roots,
            root_count,
            needs_write_barrier,
        )
    }
}

/// Process-global callback that performs a nursery allocation for the
/// currently active backend. The callback returns `GcRef(0)` (i.e.
/// null) on allocation failure so callers can fall back to a
/// non-GC allocator.
pub type AllocNurseryTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_NURSERY_TYPED: AllocNurseryTypedFn);

/// Install the active backend's nursery allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_nursery_typed(hook: Option<AllocNurseryTypedFn>) {
    ACTIVE_ALLOC_NURSERY_TYPED.set(hook);
}

/// Whether a backend owns the heap.
///
/// The allocation entry points below answer both "no backend" and "the GC
/// refused" with `GcRef(0)`, and the two are not interchangeable. Nothing owns
/// the heap during a bare unit test or the pre-`init_gc_subsystem` bootstrap,
/// so a caller's own `malloc(flavor='raw')` path *is* the whole heap there and
/// taking it keeps the object graph consistent. Once a backend is installed
/// that same fallback would leave a raw pointer in a field the collector traces
/// as a GC reference, and the caller must fail instead.
///
/// `set_gc_allocator` publishes every allocation hook in one block
/// (`register_active_hooks` in `majit-backend-dynasm/src/runner.rs` and its
/// cranelift/wasm twins),
/// so this one cell answers for all of them.
///
/// Upstream has neither state: the GC is a prebuilt constant
/// (`rpython/memory/gctransform/framework.py:254`) and a nursery that cannot
/// satisfy a request reaches `collect_and_reserve`
/// (`rpython/memory/gc/incminimark.py`), which raises MemoryError.
pub fn gc_allocator_installed() -> bool {
    ACTIVE_ALLOC_NURSERY_TYPED.get().is_some()
}

/// Whether this thread can satisfy a collecting nursery allocation.
///
/// [`gc_allocator_installed`] is process-global: installing a backend hook
/// on any thread makes it true for the whole process. A collecting nursery
/// alloc still needs this thread's collector — the `gc_sync` singleton, or a
/// box stored in this thread's TLS. `install_jitframe_gc` (and
/// `install_gc_box`) install a box once per execution thread; a thread
/// without either falls through the dynasm trampoline into
/// `standalone_alloc_*`, which panics on an uninitialized singleton.
pub type CurrentThreadHasGcBoxFn = fn() -> bool;

global_hook!(static ACTIVE_THREAD_HAS_GC_BOX: CurrentThreadHasGcBoxFn);

/// Install the active backend's "this thread holds a GC box" query.
/// Pass `None` to clear. Backends register `gc_box::present`.
pub fn set_active_thread_has_gc_box(hook: Option<CurrentThreadHasGcBoxFn>) {
    ACTIVE_THREAD_HAS_GC_BOX.set(hook);
}

pub fn current_thread_gc_available() -> bool {
    gc_sync::is_initialized() || ACTIVE_THREAD_HAS_GC_BOX.get().map(|f| f()).unwrap_or(false)
}

/// Backend-installed `raise MemoryError` for raw malloc OOM
/// (`transform.py _ll_malloc_fixedsize` / `llmemory.raw_malloc`).
/// Each backend's hook is `jit_exc_raise(memory_error_singleton_ref())`.
pub type RaiseMemoryErrorFn = fn();

global_hook!(static ACTIVE_RAISE_MEMORY_ERROR: RaiseMemoryErrorFn);

/// Install the active backend's MemoryError raise. Pass `None` to clear.
pub fn set_active_raise_memory_error(hook: Option<RaiseMemoryErrorFn>) {
    ACTIVE_RAISE_MEMORY_ERROR.set(hook);
}

/// Store the prebuilt MemoryError singleton as the pending exception.
/// No-op when no hook is installed, matching `memory_error_singleton_ref`
/// with no provider.
pub fn raise_memory_error() {
    if let Some(hook) = ACTIVE_RAISE_MEMORY_ERROR.get() {
        hook();
    }
}

static MINOR_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bumped once at the end of each minor (`minor_collection_body`).
pub fn minor_epoch() -> u64 {
    MINOR_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn bump_minor_epoch() {
    MINOR_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// What an allocation answered, with the two non-pointer states kept apart.
///
/// `malloc_fixedsize` (incminimark.py) has neither state. The GC is a
/// prebuilt constant (framework.py:254), so a route always exists, and a
/// nursery that cannot satisfy the request reaches `collect_and_reserve`
/// (incminimark.py), which raises MemoryError rather than handing a
/// null back. Both states are this port's own, and they are not
/// interchangeable:
///
/// * [`NoRoute`](Self::NoRoute) — nothing owns the heap yet: a bare unit
///   test, the pre-`init_gc_subsystem` bootstrap, or a build with no backend.
///   The caller's `malloc_raw` path *is* the whole heap there, so taking it
///   keeps the object graph consistent.
/// * [`Failed`](Self::Failed) — a GC owns the heap and could not satisfy the
///   request. `malloc_raw` is the wrong answer now: the object would hold
///   managed references the collector never traces or forwards, and its
///   missing header lets a type-id witness misread the words before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcAllocOutcome {
    Allocated(GCREF),
    Failed,
    NoRoute,
}

impl GcAllocOutcome {
    /// Classify a hook result: `None` is [`NoRoute`](Self::NoRoute),
    /// `Some(null)` is [`Failed`](Self::Failed).
    #[inline]
    pub fn from_hook(result: Option<GCREF>) -> Self {
        match result {
            Some(raw) if !raw.is_null() => Self::Allocated(raw),
            Some(_) => Self::Failed,
            None => Self::NoRoute,
        }
    }

    /// Classify what an allocation entry point in this module returned. Those
    /// answer both states with `GcRef(0)`, so the null is read against
    /// [`gc_allocator_installed`] to tell them apart.
    #[inline]
    pub fn classify(raw: GcRef) -> Self {
        if raw.0 != 0 {
            Self::Allocated(raw.0 as GCREF)
        } else if gc_allocator_installed() {
            Self::Failed
        } else {
            Self::NoRoute
        }
    }

    /// The allocated pointer, or `None` for [`NoRoute`](Self::NoRoute) so the
    /// caller takes its own non-GC path. A [`Failed`](Self::Failed) does not
    /// return: see [`gc_alloc_failed`].
    #[inline]
    pub fn allocated_or_abort(self, payload_size: usize) -> Option<GCREF> {
        match self {
            Self::Allocated(raw) => Some(raw),
            Self::Failed => gc_alloc_failed(payload_size),
            Self::NoRoute => None,
        }
    }
}

/// A GC that owns the heap could not satisfy an allocation.
///
/// `collect_and_reserve` (incminimark.py) raises MemoryError at this
/// point, so no caller of `malloc_fixedsize` observes a null. The callers here
/// return a bare pointer and run under JIT frames that cannot unwind, so the
/// failure aborts instead — the answer `alloc_typed_items_block_nursery`
/// already gives for a digit array whose allocation fails.
#[cold]
#[inline(never)]
pub fn gc_alloc_failed(payload_size: usize) -> ! {
    let layout = std::alloc::Layout::from_size_align(payload_size, std::mem::align_of::<usize>())
        .unwrap_or_else(|_| std::alloc::Layout::new::<usize>());
    std::alloc::handle_alloc_error(layout)
}

/// Allocate through the active backend's GC. Returns `GcRef(0)` when
/// no backend has installed a hook (callers treat this as a
/// null pointer and fall back to their non-GC path).
pub fn alloc_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED.get() {
        Some(_) if !gc_box_installed() => standalone_alloc_nursery_typed(type_id, payload_size),
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// The nursery bump of the active backend's GC, without a collection and
/// without an old-generation spill: `GcRef(0)` when the nursery has no room,
/// the request is large, or no direct backend GC is installed.
pub fn try_alloc_nursery_bump_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED.get() {
        Some(_) if !gc_box_installed() => {
            gc_sync::gc_op(|g| g.try_alloc_nursery_bump_typed(type_id, payload_size))
        }
        _ => GcRef(0),
    }
}

/// [`alloc_nursery_typed`] for a type that carries neither a finalizer nor the
/// weakref flag: `malloc_fast` (`framework.py:361-382`), the copy
/// `gct_fv_gc_malloc` (`framework.py`) selects for exactly that case.
///
/// Only the direct path folds the two registrations out. A per-thread GC box is
/// a backend-test configuration, so its `AllocNurseryTypedFn` keeps running the
/// general body — always a correct implementation of the fast one, just with
/// the two constant-false tests still present.
///
/// # Safety
/// `type_id` must name a type with no destructor and no weakref flag.
pub unsafe fn alloc_fast_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED.get() {
        Some(_) if !gc_box_installed() => unsafe {
            standalone_alloc_fast_nursery_typed(type_id, payload_size)
        },
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// Process-global callback for a headerless, non-collecting nursery
/// allocation. Returns `GcRef(0)` when no backend has installed a hook, so
/// the caller can keep its own non-GC path.
pub type AllocNurseryHeaderlessNoCollectFn = fn(size: usize) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_HEADERLESS_NO_COLLECT: AllocNurseryHeaderlessNoCollectFn
);

/// Install the active backend's headerless no-collect allocator callback.
/// Pass `None` to clear.
pub fn set_active_alloc_nursery_headerless_no_collect(
    hook: Option<AllocNurseryHeaderlessNoCollectFn>,
) {
    ACTIVE_ALLOC_NURSERY_HEADERLESS_NO_COLLECT.set(hook);
}

/// Allocate a headerless object through the active backend's GC without
/// letting it collect. Returns `GcRef(0)` when no backend installed a hook.
///
/// The metainterp's jitcode tracer reaches its `NEW` allocation through here:
/// an interpreter that declares `headerless_structs` owns those objects in its
/// own collected pool, so a plain host-heap block there would be invisible to
/// its collector for as long as the object stays reachable.
pub fn alloc_nursery_headerless_no_collect(size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_HEADERLESS_NO_COLLECT.get() {
        Some(f) => f(size),
        None => GcRef(0),
    }
}

/// Placement-reporting companion of [`AllocNurseryTypedFn`] for fresh-object
/// initialization. The allocation remains no-collect; the out-parameter is
/// `false` for a nursery result and `true` for an old-gen spill.
pub type AllocNurseryTypedWithPlacementFn =
    unsafe fn(type_id: u32, payload_size: usize, needs_write_barrier: *mut bool) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT:
        AllocNurseryTypedWithPlacementFn
);

pub fn set_active_alloc_nursery_typed_with_placement(
    hook: Option<AllocNurseryTypedWithPlacementFn>,
) {
    ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT.set(hook);
}

/// Allocate through the active backend's no-collect nursery allocator while
/// reporting the result placement.
///
/// # Safety
/// `needs_write_barrier` must remain a valid mutable `bool` slot until this
/// call returns.
pub unsafe fn alloc_nursery_typed_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT.get() {
        Some(f) => unsafe { f(type_id, payload_size, needs_write_barrier) },
        None => {
            // No backend installed placement reporting, so keep the
            // conservative creation barrier the sibling fallbacks report.
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// [`alloc_nursery_typed_with_placement`] for a type that carries neither a
/// finalizer nor the weakref flag: `malloc_fast` (`framework.py:361-382`).
///
/// Like [`alloc_fast_nursery_typed`], only the direct path folds the two
/// registrations out; a per-thread box keeps the general body.
///
/// # Safety
/// Same contract as [`alloc_nursery_typed_with_placement`], plus `type_id` must
/// name a type with no destructor and no weakref flag.
pub unsafe fn alloc_fast_nursery_typed_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT.get() {
        Some(_) if !gc_box_installed() => gc_sync::gc_op(|g| unsafe {
            g.try_alloc_fast_nursery_no_collect_typed_with_placement(
                type_id,
                payload_size,
                needs_write_barrier,
            )
        }),
        Some(f) => unsafe { f(type_id, payload_size, needs_write_barrier) },
        None => {
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// Process-global callback that performs a stable-address old-gen
/// allocation for the currently active backend. Used by host-side
/// allocators whose callers hold the returned pointer on the Rust
/// stack without registering it as a GC root. MiniMark's
/// old-gen is mark-sweep (non-moving), so a subsequent minor
/// collection cannot invalidate the pointer. The callback returns
/// `GcRef(0)` on allocation failure.
pub type AllocOldgenTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_OLDGEN_TYPED: AllocOldgenTypedFn);

/// Install the active backend's old-gen allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_oldgen_typed(hook: Option<AllocOldgenTypedFn>) {
    ACTIVE_ALLOC_OLDGEN_TYPED.set(hook);
}

/// Allocate a stable-address (old-gen) object through the active
/// backend's GC. Returns `GcRef(0)` when no backend has installed a hook.
pub fn alloc_oldgen_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_OLDGEN_TYPED.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// Process-global callback for [`GcAllocator::alloc_young_nonmoving_typed`]:
/// a stable address that the next minor collection frees unless something
/// reaches it. Returns `GcRef(0)` on allocation failure.
pub type AllocYoungNonmovingTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_YOUNG_NONMOVING_TYPED: AllocYoungNonmovingTypedFn);

/// Install the active backend's young non-moving allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_young_nonmoving_typed(hook: Option<AllocYoungNonmovingTypedFn>) {
    ACTIVE_ALLOC_YOUNG_NONMOVING_TYPED.set(hook);
}

/// Allocate a young, non-moving object through the active backend's GC.
/// A backend that installed no young hook answers with
/// [`alloc_oldgen_typed`]; `GcRef(0)` when no backend is installed.
pub fn alloc_young_nonmoving_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_YOUNG_NONMOVING_TYPED.get() {
        Some(f) => f(type_id, payload_size),
        None => alloc_oldgen_typed(type_id, payload_size),
    }
}

global_hook!(static ACTIVE_ALLOC_YOUNG_NONMOVING_TYPED_NO_COLLECT: AllocYoungNonmovingTypedFn);

/// Install the active backend's non-collecting young non-moving allocator
/// callback ([`GcAllocator::alloc_young_nonmoving_typed_no_collect`]). Pass
/// `None` to clear.
pub fn set_active_alloc_young_nonmoving_typed_no_collect(hook: Option<AllocYoungNonmovingTypedFn>) {
    ACTIVE_ALLOC_YOUNG_NONMOVING_TYPED_NO_COLLECT.set(hook);
}

/// [`alloc_young_nonmoving_typed`] without a collection in front of the
/// birth. A backend that installed no such hook answers with
/// [`alloc_oldgen_typed`]; `GcRef(0)` when no backend is installed.
pub fn alloc_young_nonmoving_typed_no_collect(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_YOUNG_NONMOVING_TYPED_NO_COLLECT.get() {
        Some(f) => f(type_id, payload_size),
        None => alloc_oldgen_typed(type_id, payload_size),
    }
}

/// Process-global callback for a *collecting* nursery allocation — unlike
/// [`alloc_nursery_typed`] (which the backends install as the no-collect
/// variant), this one runs a minor collection when the nursery is full instead
/// of spilling to old-gen. Only safe for callers that hold no unrooted GC
/// pointer across the allocation AND run at a JIT safepoint whose gcmap roots
/// the live set (e.g. the elidable bigint payload helpers, invoked from a
/// gcmap-carrying residual CallR). Returns `GcRef(0)` when no backend installed.
pub type AllocNurseryCollectingTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED: AllocNurseryCollectingTypedFn);

/// Install the active backend's collecting-nursery allocator callback. Pass
/// `None` to clear. Backends that do not install one leave callers to fall back
/// to the no-collect path.
pub fn set_active_alloc_nursery_collecting_typed(hook: Option<AllocNurseryCollectingTypedFn>) {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.set(hook);
}

/// Whether a backend installed the collecting nursery allocator.
pub fn alloc_nursery_collecting_typed_installed() -> bool {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.get().is_some()
}

/// Allocate through the active backend's collecting nursery allocator. Returns
/// `GcRef(0)` when no backend (or no collecting hook) is installed (callers
/// treat this as null and fall back to the no-collect path).
pub fn alloc_nursery_collecting_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// Process-global rooted companion of
/// [`AllocNurseryCollectingTypedFn`]. The caller supplies the one GC slot that
/// is live only on the native Rust stack while the allocation may collect.
pub type AllocNurseryCollectingTypedRootedFn = unsafe fn(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED:
        AllocNurseryCollectingTypedRootedFn
);

pub fn set_active_alloc_nursery_collecting_typed_rooted(
    hook: Option<AllocNurseryCollectingTypedRootedFn>,
) {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED.set(hook);
}

/// Allocate through the active rooted collecting allocator.
///
/// # Safety
/// `root` must remain a valid mutable `GcRef` slot until this call returns.
/// `needs_write_barrier` must remain a valid mutable `bool` slot.
pub unsafe fn alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED.get() {
        Some(_) if !gc_box_installed() => unsafe {
            standalone_alloc_nursery_collecting_typed_rooted(
                type_id,
                payload_size,
                root,
                needs_write_barrier,
            )
        },
        Some(f) => unsafe { f(type_id, payload_size, root, needs_write_barrier) },
        None => {
            // No backend installed placement reporting, so keep the
            // conservative creation barrier the sibling fallbacks report.
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// [`alloc_nursery_collecting_typed_rooted`] for a type that carries neither a
/// finalizer nor the weakref flag: `malloc_fast` (`framework.py:361-382`).
///
/// Translated RPython inlines this against the prebuilt GC's
/// `nursery_free` / `nursery_top`. The bump reads those same words;
/// a per-thread box (which disarms the published pair) is the cold arm.
///
/// # Safety
/// Same contract as [`alloc_nursery_collecting_typed_rooted`], plus `type_id`
/// must name a type with no destructor and no weakref flag.
#[inline(always)]
pub unsafe fn alloc_fast_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    unsafe { malloc_fast_bump(type_id, payload_size, root, 1, needs_write_barrier) }
}

/// Process-global root-span companion of
/// [`alloc_fast_nursery_collecting_typed_rooted`].
pub type AllocNurseryCollectingTypedRootsFn = unsafe fn(
    type_id: u32,
    payload_size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTS:
        AllocNurseryCollectingTypedRootsFn
);

pub fn set_active_alloc_nursery_collecting_typed_roots(
    hook: Option<AllocNurseryCollectingTypedRootsFn>,
) {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTS.set(hook);
}

/// Allocate through `malloc_fast` with an RPython `gc_push_roots` livevars span.
///
/// # Safety
/// `roots` must address `root_count` contiguous mutable [`GcRef`] slots for the
/// duration of the call.
#[inline(always)]
pub unsafe fn alloc_fast_nursery_collecting_typed_roots(
    type_id: u32,
    payload_size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    unsafe {
        malloc_fast_bump(
            type_id,
            payload_size,
            roots,
            root_count,
            needs_write_barrier,
        )
    }
}

/// Process-global callback that runs `GcAllocator::collect_generation` on the
/// active backend's GC. App-level `gc.collect(n)` reaches the live GC through
/// this trampoline, carrying the generation it was given: the backends do not
/// share one GC, so the dispatch cannot happen on `gc_sync`'s singleton.
/// Returns silently when no backend has installed a hook (callers treat it as
/// a no-op).
pub type CollectGenerationFn = fn(i64);

global_hook!(static ACTIVE_COLLECT_GENERATION: CollectGenerationFn);

/// Install the active backend's collection trampoline. Pass `None` to clear.
pub fn set_active_collect_generation(hook: Option<CollectGenerationFn>) {
    ACTIVE_COLLECT_GENERATION.set(hook);
}

/// Run `incminimark.py collect(gen)` on the active backend's GC.
/// No-op when no backend has installed a hook.
pub fn collect_generation(generation: i64) {
    if let Some(f) = ACTIVE_COLLECT_GENERATION.get() {
        f(generation);
    }
}

/// [`collect_generation`] at the generation that means "all of it". Named
/// separately because `majit-translate`'s gctransform knows this symbol.
pub fn collect_full() {
    collect_generation(2);
}

/// Active-backend trampoline for `incminimark.py collect_step`.
pub type CollectStepFn = fn() -> GcStepTransition;

global_hook!(static ACTIVE_COLLECT_STEP: CollectStepFn);

pub fn set_active_collect_step(hook: Option<CollectStepFn>) {
    ACTIVE_COLLECT_STEP.set(hook);
}

/// With no backend installed there is no state machine and nothing to step, so
/// this reports the same completed transition the trait default does
/// (`rgc.py`'s `_encode_states(1, 0)`). Reporting the starting state on
/// both sides would say a collection is still in progress, and a caller
/// stepping until [`GcStepTransition::is_done`] would never stop.
pub fn collect_step() -> GcStepTransition {
    ACTIVE_COLLECT_STEP.get().map_or(
        GcStepTransition {
            old_state: GcStepTransition::STATE_MARKING,
            new_state: GcStepTransition::STATE_SCANNING,
        },
        |step| step(),
    )
}

/// Active-backend trampoline for `rpython/rlib/rgc.py do_get_objects
/// do_get_objects`. The generation values are CPython 3.14's public
/// `gc.get_objects` convention. The backend must invoke the visitor while its
/// inspection pause is still held.
pub type GetObjectsVisitorFn = fn(GcRef);
pub type GetObjectsFn = fn(i8, GetObjectsVisitorFn);

global_hook!(static ACTIVE_GET_OBJECTS: GetObjectsFn);

pub fn set_active_get_objects(hook: Option<GetObjectsFn>) {
    ACTIVE_GET_OBJECTS.set(hook);
}

pub fn get_objects(generation: i8, visitor: GetObjectsVisitorFn) {
    if let Some(f) = ACTIVE_GET_OBJECTS.get() {
        f(generation, visitor);
    }
}

/// Active-backend trampoline for `pypy/module/gc/referents.py _list_w_obj_referents
/// _list_w_obj_referents`. Same rooting contract as [`get_objects`]: the
/// backend invokes the visitor while its inspection pause is still held.
pub type GetReferentsFn = fn(GcRef, GetObjectsVisitorFn);

global_hook!(static ACTIVE_GET_REFERENTS: GetReferentsFn);

pub fn set_active_get_referents(hook: Option<GetReferentsFn>) {
    ACTIVE_GET_REFERENTS.set(hook);
}

pub fn get_referents(obj: GcRef, visitor: GetObjectsVisitorFn) {
    if let Some(f) = ACTIVE_GET_REFERENTS.get() {
        f(obj, visitor);
    }
}

/// Backend-routed prefilter for a prompt-finalization boundary.  Unlike the
/// process-wide finalizer census, this asks only about the graph whose owning
/// edge is about to be released.
pub type SubgraphHasPendingFinalizerFn = fn(&[GcRef]) -> bool;

global_hook!(static ACTIVE_SUBGRAPH_HAS_PENDING_FINALIZER: SubgraphHasPendingFinalizerFn);

pub fn set_active_subgraph_has_pending_finalizer(hook: Option<SubgraphHasPendingFinalizerFn>) {
    ACTIVE_SUBGRAPH_HAS_PENDING_FINALIZER.set(hook);
}

pub fn subgraph_has_pending_finalizer(roots: &[GcRef]) -> bool {
    match ACTIVE_SUBGRAPH_HAS_PENDING_FINALIZER.get() {
        Some(f) => f(roots),
        None if gc_sync::is_initialized() => {
            gc_sync::gc_op(|gc| gc.subgraph_has_pending_finalizer(roots))
        }
        // No collector cannot own or finalize any value in the graph.
        None => false,
    }
}

/// Active-backend trampoline for `gc.is_tracked`: whether the collector
/// traverses references out of the object.
pub type IsTrackedFn = fn(GcRef) -> bool;

global_hook!(static ACTIVE_IS_TRACKED: IsTrackedFn);

pub fn set_active_is_tracked(hook: Option<IsTrackedFn>) {
    ACTIVE_IS_TRACKED.set(hook);
}

pub fn is_tracked(obj: GcRef) -> bool {
    match ACTIVE_IS_TRACKED.get() {
        Some(f) => f(obj),
        None => false,
    }
}

/// Active-backend hooks for `rgc`'s temporary traversal flags.
pub type GetGcFlagFn = fn(GcRef) -> bool;
pub type ToggleGcFlagFn = fn(GcRef);

global_hook!(static ACTIVE_GET_GCFLAG_EXTRA: GetGcFlagFn);
global_hook!(static ACTIVE_TOGGLE_GCFLAG_EXTRA: ToggleGcFlagFn);
global_hook!(static ACTIVE_GET_GCFLAG_DUMMY: GetGcFlagFn);

pub fn set_active_gcflag_hooks(
    get_extra: Option<GetGcFlagFn>,
    toggle_extra: Option<ToggleGcFlagFn>,
    get_dummy: Option<GetGcFlagFn>,
) {
    ACTIVE_GET_GCFLAG_EXTRA.set(get_extra);
    ACTIVE_TOGGLE_GCFLAG_EXTRA.set(toggle_extra);
    ACTIVE_GET_GCFLAG_DUMMY.set(get_dummy);
}

/// `rgc.has_gcflag_extra`: IncrementalMiniMark exposes `GCFLAG_EXTRA`.
pub const fn has_gcflag_extra() -> bool {
    true
}

pub fn get_gcflag_extra(obj: GcRef) -> bool {
    ACTIVE_GET_GCFLAG_EXTRA.get().is_some_and(|f| f(obj))
}

pub fn toggle_gcflag_extra(obj: GcRef) {
    let toggle = ACTIVE_TOGGLE_GCFLAG_EXTRA
        .get()
        .expect("toggle_gcflag_extra called without an active collector");
    toggle(obj);
}

pub fn get_gcflag_dummy(obj: GcRef) -> bool {
    ACTIVE_GET_GCFLAG_DUMMY.get().is_some_and(|f| f(obj))
}

/// Translated `rgc.hide_nonmovable_gcref`: after the non-moving assertion,
/// the representation change is just GCREF-to-address.
pub fn hide_nonmovable_gcref(obj: GcRef) -> usize {
    assert!(obj.is_null() || !can_move(obj));
    obj.0
}

/// Translated `rgc.reveal_gcref`: address-to-GCREF inverse.
pub const fn reveal_gcref(addr: usize) -> GcRef {
    GcRef(addr)
}

/// Active-backend trampolines for `inspector.py:get_rpy_memory_usage` and
/// `get_rpy_type_index`.  The `Option` is the Rust spelling of upstream's
/// negative result for a missing collector operation.
pub type GetRpyIntrospectionFn = fn(GcRef) -> Option<usize>;

global_hook!(static ACTIVE_GET_RPY_MEMORY_USAGE: GetRpyIntrospectionFn);
global_hook!(static ACTIVE_GET_RPY_TYPE_INDEX: GetRpyIntrospectionFn);

pub fn set_active_get_rpy_memory_usage(hook: Option<GetRpyIntrospectionFn>) {
    ACTIVE_GET_RPY_MEMORY_USAGE.set(hook);
}

pub fn set_active_get_rpy_type_index(hook: Option<GetRpyIntrospectionFn>) {
    ACTIVE_GET_RPY_TYPE_INDEX.set(hook);
}

pub fn get_rpy_memory_usage(obj: GcRef) -> Option<usize> {
    ACTIVE_GET_RPY_MEMORY_USAGE.get().and_then(|f| f(obj))
}

pub fn get_rpy_type_index(obj: GcRef) -> Option<usize> {
    ACTIVE_GET_RPY_TYPE_INDEX.get().and_then(|f| f(obj))
}

pub type GetRpyRootsFn = fn(GetObjectsVisitorFn) -> bool;
pub type GetRpyReferentsFn = fn(GcRef, GetObjectsVisitorFn) -> bool;
pub type IsAppLevelObjectFn = fn(GcRef) -> bool;

global_hook!(static ACTIVE_GET_RPY_ROOTS: GetRpyRootsFn);
global_hook!(static ACTIVE_GET_RPY_REFERENTS: GetRpyReferentsFn);
global_hook!(static ACTIVE_IS_APP_LEVEL_OBJECT: IsAppLevelObjectFn);

pub fn set_active_get_rpy_roots(hook: Option<GetRpyRootsFn>) {
    ACTIVE_GET_RPY_ROOTS.set(hook);
}

pub fn set_active_get_rpy_referents(hook: Option<GetRpyReferentsFn>) {
    ACTIVE_GET_RPY_REFERENTS.set(hook);
}

pub fn set_active_is_app_level_object(hook: Option<IsAppLevelObjectFn>) {
    ACTIVE_IS_APP_LEVEL_OBJECT.set(hook);
}

pub fn get_rpy_roots(visitor: GetObjectsVisitorFn) -> bool {
    ACTIVE_GET_RPY_ROOTS.get().is_some_and(|f| f(visitor))
}

pub fn get_rpy_referents(obj: GcRef, visitor: GetObjectsVisitorFn) -> bool {
    ACTIVE_GET_RPY_REFERENTS
        .get()
        .is_some_and(|f| f(obj, visitor))
}

pub fn is_app_level_object(obj: GcRef) -> bool {
    ACTIVE_IS_APP_LEVEL_OBJECT.get().is_some_and(|f| f(obj))
}

pub type DumpRpyHeapFn = fn(i32) -> Result<bool, i32>;
pub type HeapDumpWriteFn = fn(i32, &[u8]) -> Result<isize, i32>;
pub type GetTypeidsTextFn = fn() -> Option<Vec<u8>>;
pub type GetTypeidsListFn = fn() -> Option<Vec<usize>>;

global_hook!(static ACTIVE_DUMP_RPY_HEAP: DumpRpyHeapFn);
global_hook!(static HEAP_DUMP_WRITE: HeapDumpWriteFn);
global_hook!(static ACTIVE_GET_TYPEIDS_TEXT: GetTypeidsTextFn);
global_hook!(static ACTIVE_GET_TYPEIDS_LIST: GetTypeidsListFn);

pub fn set_active_dump_rpy_heap(hook: Option<DumpRpyHeapFn>) {
    ACTIVE_DUMP_RPY_HEAP.set(hook);
}

/// Install the host write used by `inspector.py HeapDumper.flush`.
/// Sandboxed interpreters use this to translate guest descriptors through
/// their host seam; when absent, the collector retains its native raw write.
pub fn set_heap_dump_write(hook: Option<HeapDumpWriteFn>) {
    HEAP_DUMP_WRITE.set(hook);
}

pub(crate) fn try_heap_dump_write(fd: i32, bytes: &[u8]) -> Option<Result<isize, i32>> {
    HEAP_DUMP_WRITE.get().map(|write| write(fd, bytes))
}

pub fn set_active_get_typeids_text(hook: Option<GetTypeidsTextFn>) {
    ACTIVE_GET_TYPEIDS_TEXT.set(hook);
}

pub fn set_active_get_typeids_list(hook: Option<GetTypeidsListFn>) {
    ACTIVE_GET_TYPEIDS_LIST.set(hook);
}

pub fn dump_rpy_heap(fd: i32) -> Result<bool, i32> {
    ACTIVE_DUMP_RPY_HEAP
        .get()
        .map_or(Ok(false), |hook| hook(fd))
}

pub fn get_typeids_text() -> Option<Vec<u8>> {
    ACTIVE_GET_TYPEIDS_TEXT.get().and_then(|hook| hook())
}

pub fn get_typeids_list() -> Option<Vec<usize>> {
    ACTIVE_GET_TYPEIDS_LIST.get().and_then(|hook| hook())
}

/// Active-backend trampolines for `rgc.add_memory_pressure` and
/// `inspector.count_memory_pressure`. The object-bearing form is retained for
/// translated callers even though `__pypy__.add_memory_pressure` passes NULL.
pub type AddMemoryPressureFn = fn(isize, GcRef);
pub type TotalMemoryPressureFn = fn() -> isize;

global_hook!(static ACTIVE_ADD_MEMORY_PRESSURE: AddMemoryPressureFn);
global_hook!(static ACTIVE_TOTAL_MEMORY_PRESSURE: TotalMemoryPressureFn);

pub fn set_active_add_memory_pressure(hook: Option<AddMemoryPressureFn>) {
    ACTIVE_ADD_MEMORY_PRESSURE.set(hook);
}

pub fn set_active_total_memory_pressure(hook: Option<TotalMemoryPressureFn>) {
    ACTIVE_TOTAL_MEMORY_PRESSURE.set(hook);
}

pub fn add_memory_pressure(size: isize, object: GcRef) {
    if let Some(hook) = ACTIVE_ADD_MEMORY_PRESSURE.get() {
        hook(size, object);
    }
}

/// Active-backend trampoline for `incminimark.external_malloc`'s
/// `threshold_reached` collection (`force_enabled=False`).
pub type MaybeCollectForExternalMallocFn = fn(usize) -> bool;

global_hook!(
    static ACTIVE_MAYBE_COLLECT_FOR_EXTERNAL_MALLOC: MaybeCollectForExternalMallocFn
);

pub fn set_active_maybe_collect_for_external_malloc(hook: Option<MaybeCollectForExternalMallocFn>) {
    ACTIVE_MAYBE_COLLECT_FOR_EXTERNAL_MALLOC.set(hook);
}

pub fn maybe_collect_for_external_malloc(totalsize: usize) -> bool {
    ACTIVE_MAYBE_COLLECT_FOR_EXTERNAL_MALLOC
        .get()
        .is_some_and(|hook| hook(totalsize))
}

pub fn total_memory_pressure() -> isize {
    ACTIVE_TOTAL_MEMORY_PRESSURE.get().map_or(0, |hook| hook())
}

/// Process-global callback running a non-moving old-gen-only major collection
/// (`GcAllocator::collect_oldgen_nonmoving`). The interpreter GC safepoint
/// reaches it to reclaim stable-allocated interp int/float without moving the
/// nursery — so it can fire under an active JIT (nursery non-empty), unlike
/// the moving `collect_full`. No-op when no backend is installed.
pub type CollectOldgenFn = fn();

global_hook!(static ACTIVE_COLLECT_OLDGEN: CollectOldgenFn);

/// Install the active backend's non-moving-major trampoline. Pass `None` to
/// clear.
pub fn set_active_collect_oldgen(hook: Option<CollectOldgenFn>) {
    ACTIVE_COLLECT_OLDGEN.set(hook);
}

/// Trigger a non-moving old-gen-only major collection on the active backend's
/// GC. No-op when no backend has installed a hook.
pub fn collect_oldgen_nonmoving() {
    if let Some(f) = ACTIVE_COLLECT_OLDGEN.get() {
        f();
    }
}

/// Process-global callback reporting the active GC's `heap_byte_stats`
/// (`(oldgen_total, nursery_used)`). Diagnostic: it lets a host runner split
/// GC-retained memory from host-heap growth. Deciding when to collect is not
/// among its uses — that question is [`active_major_threshold_reached`], which
/// answers it from the collector's own threshold rather than from a number a
/// caller would have to compare against a threshold of its own.
pub type HeapStatsFn = fn() -> (usize, usize);

global_hook!(static ACTIVE_HEAP_STATS: HeapStatsFn);

/// Install the active backend's `heap_byte_stats` trampoline.
pub fn set_active_heap_stats(hook: Option<HeapStatsFn>) {
    ACTIVE_HEAP_STATS.set(hook);
}

/// Report `(oldgen_total, nursery_used)` from the active backend's GC.
/// `(0, 0)` when no backend has installed a hook.
pub fn active_heap_stats() -> (usize, usize) {
    match ACTIVE_HEAP_STATS.get() {
        Some(f) => f(),
        None => (0, 0),
    }
}

/// Active-backend trampoline for `incminimark.py:get_stats`.
pub type GcMemoryStatsFn = fn() -> GcMemoryStats;

global_hook!(static ACTIVE_GC_MEMORY_STATS: GcMemoryStatsFn);

pub fn set_active_gc_memory_stats(hook: Option<GcMemoryStatsFn>) {
    ACTIVE_GC_MEMORY_STATS.set(hook);
}

pub fn active_gc_memory_stats() -> GcMemoryStats {
    match ACTIVE_GC_MEMORY_STATS.get() {
        Some(f) => f(),
        None => GcMemoryStats::default(),
    }
}

/// Active JIT CPU's `AsmMemoryManager.get_stats()` pair, exposed here so the
/// interpreter-level `gc` module does not acquire a dependency on a concrete
/// machine-code backend (`pypy/module/gc/referents.py:_get_stats`).
pub type JitBackendMemoryStatsFn = fn() -> (usize, usize);

global_hook!(static ACTIVE_JIT_BACKEND_MEMORY_STATS: JitBackendMemoryStatsFn);

pub fn set_active_jit_backend_memory_stats(hook: Option<JitBackendMemoryStatsFn>) {
    ACTIVE_JIT_BACKEND_MEMORY_STATS.set(hook);
}

pub fn active_jit_backend_memory_stats() -> (usize, usize) {
    ACTIVE_JIT_BACKEND_MEMORY_STATS
        .get()
        .map_or((0, 0), |hook| hook())
}

/// Process-global callback reporting the active GC's `major_threshold_reached`
/// (incminimark.py threshold_reached). The interpreter GC safepoint
/// (`pyre_object::gc_interp`) collects when this says the collector wants a
/// major, instead of keeping a second, poorer model of heap growth beside the
/// collector's own.
pub type MajorThresholdReachedFn = fn() -> bool;

global_hook!(static ACTIVE_MAJOR_THRESHOLD_REACHED: MajorThresholdReachedFn);

/// Install the active backend's `major_threshold_reached` trampoline.
pub fn set_active_major_threshold_reached(hook: Option<MajorThresholdReachedFn>) {
    ACTIVE_MAJOR_THRESHOLD_REACHED.set(hook);
}

/// Whether the active backend's GC has reached its next-major threshold.
/// `false` when no backend has installed a hook, so a caller with no collector
/// behind it never collects.
pub fn active_major_threshold_reached() -> bool {
    match ACTIVE_MAJOR_THRESHOLD_REACHED.get() {
        Some(f) => f(),
        None => false,
    }
}

/// Process-global callback for [`GcAllocator::minor_collections_since_major`],
/// installed by whichever backend owns the GC. The interpreter's `gc` module
/// asks through this rather than through `gc_sync`'s singleton, because the
/// wasm backend keeps its GC somewhere that singleton does not reach.
pub type MinorCollectionsSinceMajorFn = fn() -> usize;

global_hook!(static ACTIVE_MINOR_COLLECTIONS_SINCE_MAJOR: MinorCollectionsSinceMajorFn);

/// Install the minors-since-major callback for the active backend.
pub fn set_active_minor_collections_since_major(hook: Option<MinorCollectionsSinceMajorFn>) {
    ACTIVE_MINOR_COLLECTIONS_SINCE_MAJOR.set(hook);
}

/// Minor collections the active backend's GC has run since its last major.
/// `0` when no backend has installed a hook, which is also the truthful
/// answer for a process that has collected nothing.
pub fn active_minor_collections_since_major() -> usize {
    match ACTIVE_MINOR_COLLECTIONS_SINCE_MAJOR.get() {
        Some(f) => f(),
        None => 0,
    }
}

/// Process-global callback that reports whether a raw address is owned
/// by the active backend's GC heap. Used by host-side allocators
/// (`pyre-object`'s `dealloc_items_block`) to discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks during the L1/L2 stepping-stone window:
/// `dealloc` must early-return for GC-managed pointers (the GC
/// sweeps them) and fall through to `std::alloc::dealloc` for
/// `std::alloc`-allocated ones.
pub type GcOwnsObjectFn = fn(addr: usize) -> bool;
pub type GcIsNurseryObjectFn = fn(addr: usize) -> bool;
/// `llop.shrink_array` — see [`GcAllocator::shrink_array`].
pub type GcShrinkArrayFn = fn(addr: usize, smaller_length: usize) -> bool;
/// Read-only TYPE_INFO query used by the four-argument `ll_arraymove`
/// residual target.
pub type GcVarSizeLayoutFn = fn(addr: usize) -> Option<GcVarSizeLayout>;

global_hook!(static ACTIVE_GC_OWNS_OBJECT: GcOwnsObjectFn);
global_hook!(static ACTIVE_GC_IS_NURSERY_OBJECT: GcIsNurseryObjectFn);
global_hook!(static ACTIVE_GC_SHRINK_ARRAY: GcShrinkArrayFn);
global_hook!(static ACTIVE_GC_VARSIZE_LAYOUT: GcVarSizeLayoutFn);

/// Install the active backend's `is_managed_heap_object` trampoline.
pub fn set_active_gc_owns_object(hook: Option<GcOwnsObjectFn>) {
    ACTIVE_GC_OWNS_OBJECT.set(hook);
}

/// Install the active backend's nursery-membership predicate.
pub fn set_active_gc_is_nursery_object(hook: Option<GcIsNurseryObjectFn>) {
    ACTIVE_GC_IS_NURSERY_OBJECT.set(hook);
}

/// Install the active backend's in-place array shrink.
pub fn set_active_gc_shrink_array(hook: Option<GcShrinkArrayFn>) {
    ACTIVE_GC_SHRINK_ARRAY.set(hook);
}

/// Install the active backend's registered-varsize-layout query.
pub fn set_active_gc_varsize_layout(hook: Option<GcVarSizeLayoutFn>) {
    ACTIVE_GC_VARSIZE_LAYOUT.set(hook);
}

/// minimark.py `id_or_identityhash` hook.
pub type GcIdOrIdentityHashFn = majit_ir::value::GcIdOrIdentityHashFn;

pub fn set_active_gc_id_or_identityhash(hook: Option<GcIdOrIdentityHashFn>) {
    majit_ir::set_gc_id_or_identityhash(hook);
}

/// Live-address hook. `intern` follows a nursery forwarding stub through
/// this so a from-space intern cannot `stamp_reuse` dest with a gap corpse.
pub type GcCurrentObjectAddressFn = majit_ir::value::GcCurrentObjectAddressFn;

pub fn set_active_gc_current_object_address(hook: Option<GcCurrentObjectAddressFn>) {
    majit_ir::set_gc_current_object_address(hook);
}

/// Return a GC-move-stable address for identity hashing.
/// Falls back to `addr` when no backend is installed.
#[inline]
pub fn gc_id_or_identityhash(addr: usize) -> usize {
    majit_ir::gc_id_or_identityhash(addr)
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no backend has installed a hook —
/// callers treat that as "no GC owns this pointer" and fall through
/// to their non-GC dealloc path.
pub fn gc_owns_object(addr: usize) -> bool {
    match ACTIVE_GC_OWNS_OBJECT.get() {
        Some(f) => f(addr),
        None => false,
    }
}

/// Whether a backend has installed [`gc_owns_object`].
///
/// [`gc_owns_object`] is `false` both when no collector exists and when a
/// collector exists but does not own `addr`. Callers that write into host
/// frames (`PyFrame::new` in tests) need the first case to proceed and the
/// second to refuse.
pub fn gc_owns_object_hook_installed() -> bool {
    ACTIVE_GC_OWNS_OBJECT.get().is_some()
}

/// `llop.shrink_array(Bool, p, smallerlength)`: record that the varsize object
/// at `addr` is shorter than it was, in place.
///
/// `false` when no backend has installed a hook, or when the object is one its
/// GC declines to resize.  `rgc.ll_shrink_array` treats that answer as
/// "allocate a smaller object and copy into it" (`rgc.py`), and so must
/// every caller here.
pub fn gc_shrink_array(addr: usize, smaller_length: usize) -> bool {
    // A residual call copies its argument out of the jitframe before it enters
    // host Rust.  A collection which ran at that boundary rewrites the
    // jitframe/root slot but not the ABI copy, so follow the forwarding word
    // before the backend reads the object's header.  RPython's GC transform
    // makes the equivalent reload around `rgc.ll_shrink_array` automatically.
    let addr = gc_current_object_address(addr);
    match ACTIVE_GC_SHRINK_ARRAY.get() {
        Some(f) => f(addr, smaller_length),
        None => false,
    }
}

/// Return the active collector's existing TYPE_INFO layout for `addr`.
///
/// `None` means either that `addr` is not a registered variable-size GC
/// object, or that no backend has installed the query — the same answer
/// [`gc_owns_object`] gives for an address no GC owns.  A caller that needs
/// a layout regardless supplies its own, the way `jit_ll_arraymove` falls
/// back to the array token of the block it allocates.
pub fn gc_varsize_layout(addr: usize) -> Option<GcVarSizeLayout> {
    if addr == 0 {
        return None;
    }
    ACTIVE_GC_VARSIZE_LAYOUT.get().and_then(|f| f(addr))
}

/// Whether the finalizer for the object at `addr` has already run.
///
/// `false` for an address the active backend's heap does not own, which is the
/// answer [`GcAllocator::register_finalizer`] already implies for one: it
/// declines to register a finalizer for such a pointer at all.
pub fn gc_finalizer_has_run(addr: usize) -> bool {
    gc_owns_object(addr) && unsafe { (*header::header_of(addr)).has_flag(GcFlags::FINALIZER_RUN) }
}

/// Record that the finalizer for the object at `addr` has run.
pub fn gc_mark_finalizer_run(addr: usize) {
    if gc_owns_object(addr) {
        unsafe { (*header::header_of(addr)).set_flag(GcFlags::FINALIZER_RUN) };
    }
}

/// Published `[start, end)` of the `gc_sync` singleton's nursery, plus the
/// tagged-pointer setting `is_valid_gc_object` consults.
///
/// `is_nursery_object_start` is `addr != 0 && !tagged && nursery.contains(addr)`
/// — three loads and two compares against fields that do not move for the life
/// of the GC instance (`reset` rewinds `free`, never `size`; the one writer of
/// `start` is `debug_rotate`, and an allocator that rotates answers
/// `nursery_bounds` with `None` rather than publish a range a minor collection
/// will move).  Reaching
/// them through the hook chain instead costs a thread-local lookup, a `RefCell`
/// borrow and a trait-object dispatch on every root pin and every root reload,
/// which is where the query is actually hot.  Publishing them lets the common
/// process shape — one singleton, no per-thread allocator box — answer inline.
///
/// Armed only for the singleton.  `install_gc_box` gives a thread its own
/// allocator with its own nursery, which a single published pair cannot
/// describe, so that path disarms this permanently
/// ([`disarm_published_nursery`]).
///
/// One object rather than five statics: upstream answers `is_in_nursery` from
/// `self.nursery` and `self.nursery_top`, two adjacent fields of the GC, so
/// the whole query is one address and two loads.  Five separate statics cost
/// the query one address materialization each, and can spread the answer over
/// as many cache lines.
#[repr(C)]
struct PublishedNursery {
    start: std::sync::atomic::AtomicUsize,
    end: std::sync::atomic::AtomicUsize,
    tagged: std::sync::atomic::AtomicBool,
    armed: std::sync::atomic::AtomicBool,
    refused: std::sync::atomic::AtomicBool,
}

static PUBLISHED_NURSERY: PublishedNursery = PublishedNursery {
    start: std::sync::atomic::AtomicUsize::new(0),
    end: std::sync::atomic::AtomicUsize::new(0),
    tagged: std::sync::atomic::AtomicBool::new(false),
    armed: std::sync::atomic::AtomicBool::new(false),
    refused: std::sync::atomic::AtomicBool::new(false),
};

/// Field addresses of `nursery_free` / `nursery_top` and the nursery
/// size cap, so `malloc_fast` can bump without `gc_op`.
///
/// `incminimark.py` `malloc_fixedsize` / `malloc_varsize` load those
/// fields off the GC object; `framework.py` inlines a copy as
/// `malloc_fast`. The Box that holds `NurseryPtrs` is stable for the
/// singleton's life, including `debug_rotate` (only the pointer
/// *values* change).
struct PublishedBump {
    free_addr: std::sync::atomic::AtomicUsize,
    top_addr: std::sync::atomic::AtomicUsize,
    large: std::sync::atomic::AtomicUsize,
}

static PUBLISHED_BUMP: PublishedBump = PublishedBump {
    free_addr: std::sync::atomic::AtomicUsize::new(0),
    top_addr: std::sync::atomic::AtomicUsize::new(0),
    large: std::sync::atomic::AtomicUsize::new(0),
};

/// Publish the singleton's nursery bounds.  A `None` from `nursery_bounds`
/// (a stub allocator, or one whose nursery is not a fixed range) leaves the
/// fast path disarmed.
///
/// The disarm-store-rearm here is not atomic against a concurrent query, so a
/// reader can pair a stale "armed" with the incoming bounds.  Both writers are
/// the singleton installers, which run before the GC has a mutator to answer
/// for; the hook path they replace is no more synchronized, since the swap
/// behind it is not atomic against a query either.
pub fn publish_singleton_nursery(gc: &dyn GcAllocator) {
    let n = &PUBLISHED_NURSERY;
    n.armed.store(false, std::sync::atomic::Ordering::Release);
    if n.refused.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    if let Some((free, top, large)) = gc.malloc_fast_bump_addrs() {
        PUBLISHED_BUMP
            .free_addr
            .store(free, std::sync::atomic::Ordering::Relaxed);
        PUBLISHED_BUMP
            .top_addr
            .store(top, std::sync::atomic::Ordering::Relaxed);
        PUBLISHED_BUMP
            .large
            .store(large, std::sync::atomic::Ordering::Relaxed);
    }
    let Some((start, end)) = gc.nursery_bounds() else {
        return;
    };
    n.start.store(start, std::sync::atomic::Ordering::Relaxed);
    n.end.store(end, std::sync::atomic::Ordering::Relaxed);
    n.tagged
        .store(gc.taggedpointers(), std::sync::atomic::Ordering::Relaxed);
    n.armed.store(true, std::sync::atomic::Ordering::Release);
}

/// Permanently retire the published fast path: a per-thread allocator box is
/// in play, so "the nursery" is no longer a single process-wide range.
pub fn disarm_published_nursery() {
    PUBLISHED_NURSERY
        .refused
        .store(true, std::sync::atomic::Ordering::Release);
    PUBLISHED_NURSERY
        .armed
        .store(false, std::sync::atomic::Ordering::Release);
    PUBLISHED_BUMP
        .free_addr
        .store(0, std::sync::atomic::Ordering::Release);
    PUBLISHED_BUMP
        .top_addr
        .store(0, std::sync::atomic::Ordering::Release);
    PUBLISHED_BUMP
        .large
        .store(0, std::sync::atomic::Ordering::Release);
}

/// Whether `addr` is a live object in the active backend's nursery.
///
/// `#[inline]`: the armed path is a two-word range compare, and the callers
/// that matter are the per-pin / per-barrier queries next door.
#[inline]
pub fn gc_is_nursery_object(addr: usize) -> bool {
    let n = &PUBLISHED_NURSERY;
    if n.armed.load(std::sync::atomic::Ordering::Acquire) {
        // `is_nursery_object_start` = `is_valid_gc_object && is_in_nursery`,
        // with the range test first because it is what answers almost every
        // query.  It subsumes the null half of `is_valid_gc_object`: a null
        // is below any published `start`, and a `start` left at 0 by an
        // allocator with no fixed range leaves `end` at 0 too, which rejects
        // everything.  The tagged half cannot be folded in — an odd word can
        // land inside the range by coincidence — but it rides behind the
        // range test rather than ahead of it.
        return addr >= n.start.load(std::sync::atomic::Ordering::Relaxed)
            && addr < n.end.load(std::sync::atomic::Ordering::Relaxed)
            && !(n.tagged.load(std::sync::atomic::Ordering::Relaxed) && addr & 1 == 1);
    }
    unpublished_is_nursery_object(addr)
}

/// [`gc_is_nursery_object`] before the singleton publishes its nursery, and
/// for a process whose per-thread allocator boxes retired the published range.
#[cold]
#[inline(never)]
fn unpublished_is_nursery_object(addr: usize) -> bool {
    match ACTIVE_GC_IS_NURSERY_OBJECT.get() {
        Some(f) => f(addr),
        None if gc_sync::is_initialized() => {
            gc_sync::gc_query_reentrant(|gc| gc.is_nursery_object(addr))
        }
        None => false,
    }
}

/// Return the current address for a managed object without treating it as a
/// root. During a minor collection this follows an already-installed nursery
/// forwarding pointer; otherwise it returns `addr` unchanged.
#[inline]
pub fn gc_current_object_address(addr: usize) -> usize {
    // Only a nursery address can carry a forwarding stub: `_trace_drag_out`
    // installs one at the young object's old address, and the major collection
    // is mark-and-sweep, so nothing outside the nursery ever moves.  The
    // nursery test is a two-word range compare, where the full ownership query
    // also walks the old-generation arena index and the rawmalloc set — and
    // every root pin and every root reload asks this question.
    if addr == 0 || !gc_is_nursery_object(addr) {
        return addr;
    }
    let hdr = unsafe { header::header_of(addr) };
    if unsafe { (*hdr).is_forwarded() } {
        unsafe { header::GcHeader::forwarding_address(hdr) }
    } else {
        addr
    }
}

/// The published nursery window `[start, end)` and the tagged-pointer setting,
/// or `None` when the published fast path is disarmed and the hook chain has
/// to answer per address.
///
/// [`gc_is_nursery_object`] reloads all four published words for every address
/// it is handed, and [`gc_sync::foreign_mutator_seen`] is a second ordered
/// load next to it.  A caller normalizing a *run* of shadow-stack slots asks
/// the identical question once per slot, so `n` livevars pay `2n` ordered
/// loads to consult operands that are the same for the whole run.  Reading the
/// window once and testing the range inline turns the per-slot cost into a
/// compare pair.
#[inline]
pub fn published_nursery_window() -> Option<(usize, usize, bool)> {
    let n = &PUBLISHED_NURSERY;
    if !n.armed.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    Some((
        n.start.load(std::sync::atomic::Ordering::Relaxed),
        n.end.load(std::sync::atomic::Ordering::Relaxed),
        n.tagged.load(std::sync::atomic::Ordering::Relaxed),
    ))
}

/// [`gc_current_object_address`] against a window a caller already read with
/// [`published_nursery_window`].
///
/// The range test is the whole of `is_nursery_object_start` for the armed
/// path: `start` is a live mapping, so a null address is below it and needs no
/// separate test, and the tagged case only has to reject an odd word.
#[inline]
pub fn gc_current_object_address_in_window(
    addr: usize,
    start: usize,
    end: usize,
    tagged: bool,
) -> usize {
    if addr < start || addr >= end || (tagged && addr & 1 == 1) {
        return addr;
    }
    let hdr = unsafe { header::header_of(addr) };
    if unsafe { (*hdr).is_forwarded() } {
        unsafe { header::GcHeader::forwarding_address(hdr) }
    } else {
        addr
    }
}

/// Process-global callbacks for registering/removing a Rust-stack slot
/// as a GC root with the currently active backend. Used by host-side
/// allocators whose callers need to keep a
/// just-allocated nursery pointer alive across a subsequent
/// potentially-collecting allocation.
///
/// RPython accomplishes the same thing automatically via its GC
/// transform pass (shadowstack save/restore around safepoints). pyre
/// lacks that pass, so root registration is explicit at the call
/// site. This is a documented TODO.
pub type AddRootFn = unsafe fn(slot: *mut GcRef);
pub type RemoveRootFn = fn(slot: *mut GcRef);

global_hook!(static ACTIVE_ADD_ROOT: AddRootFn);
global_hook!(static ACTIVE_REMOVE_ROOT: RemoveRootFn);

/// Install the active backend's root-register callbacks. Pass `None`
/// to clear.
pub fn set_active_root_hooks(add: Option<AddRootFn>, remove: Option<RemoveRootFn>) {
    if add.is_some() {
        // Publish remove before add so every newly registered root can be removed.
        ACTIVE_REMOVE_ROOT.set(remove);
        ACTIVE_ADD_ROOT.set(add);
    } else {
        // Withdraw add before remove so no root can be registered without removal.
        ACTIVE_ADD_ROOT.set(add);
        ACTIVE_REMOVE_ROOT.set(remove);
    }
}

/// Register a stack slot as a GC root with the active backend. No-op
/// when no backend has installed a hook.
///
/// Returns whether the hook ran, so a guard that pairs this with
/// [`gc_remove_root`] on drop can skip the removal of a root it never
/// registered.
///
/// # Safety
/// The caller must ensure the slot remains valid until
/// [`gc_remove_root`] is called with the same pointer.
pub unsafe fn gc_add_root(slot: *mut GcRef) -> bool {
    // A shadow-stack/root-map entry names a stationary slot whose contents
    // the collector rewrites after moving its referent.  An address inside
    // the nursery moves with its owner and becomes from-space after the first
    // minor collection; registering it would make the root map itself stale.
    // RPython's shadow-stack transform only assigns roots to fixed frame
    // slots, while fields of GC objects are reached by tracing their owner.
    if gc_is_nursery_object(slot as usize) {
        return false;
    }
    match ACTIVE_ADD_ROOT.get() {
        Some(f) => {
            unsafe { f(slot) };
            true
        }
        None => false,
    }
}

/// Remove a previously-registered root slot from the active backend.
/// No-op when no backend has installed a hook. Returns whether the hook ran.
pub fn gc_remove_root(slot: *mut GcRef) -> bool {
    match ACTIVE_REMOVE_ROOT.get() {
        Some(f) => {
            f(slot);
            true
        }
        None => false,
    }
}

/// rgc.py `FinalizerQueue.register_finalizer` - routed to the active backend GC.
pub type RegisterFinalizerFn = fn(fq_index: usize, obj: GcRef);

/// Pointer to the active collector's `FinalizerQueueTable`. Null means fall
/// through to the `gc_sync` singleton handle, then to the host-heap descr's
/// table. The getter must not borrow the collector (`framework.py ll_next_dead`
/// closes over its own `AddressDeque`).
pub type ActiveFinalizerQueuesFn = fn() -> *const FinalizerQueueTable;

global_hook!(static ACTIVE_REGISTER_FINALIZER: RegisterFinalizerFn);
global_hook!(static ACTIVE_FINALIZER_QUEUES: ActiveFinalizerQueuesFn);

/// Boehm/host-heap descr's `finalizer_queue_indexes`.
/// `boehm.py BoehmGCTransformer.get_finalizer_queue_index` owns one table
/// on the transformer; `gc.py get_ll_description(None)` selects that descr
/// when no framework collector is installed. One such descr per process.
/// A boxed collector or the singleton still uses only its own table.
static HOST_HEAP_FINALIZER_QUEUES: std::sync::LazyLock<FinalizerQueueTable> =
    std::sync::LazyLock::new(FinalizerQueueTable::new);

/// Install the active backend's `gc_fq_register` trampoline. Pass `None` to
/// clear. `get_finalizer_queue_index` / `gc_fq_next_dead` use
/// [`set_active_finalizer_queue_table`] and never borrow the collector.
pub fn set_active_finalizer_hooks(register: Option<RegisterFinalizerFn>) {
    ACTIVE_REGISTER_FINALIZER.set(register);
}

/// Install a getter for the active collector's `finalizer_handlers`. Pass
/// `None` to clear. A getter that returns null falls through to the
/// `gc_sync` singleton table handle, then to the host-heap descr's table.
pub fn set_active_finalizer_queue_table(getter: Option<ActiveFinalizerQueuesFn>) {
    ACTIVE_FINALIZER_QUEUES.set(getter);
}

fn active_finalizer_queues() -> &'static FinalizerQueueTable {
    if let Some(getter) = ACTIVE_FINALIZER_QUEUES.get() {
        let p = getter();
        if !p.is_null() {
            // SAFETY: the getter returns the table published next to the
            // active collector (backend TLS `Arc` clone, or null to fall
            // through). The `Arc` keeps the table alive for the collector's
            // lifetime.
            return unsafe { &*p };
        }
    }
    gc_sync::singleton_finalizer_queues().unwrap_or(&HOST_HEAP_FINALIZER_QUEUES)
}

/// framework.py `get_finalizer_queue_index` — allocate or reuse the handler
/// index for this FinalizerQueue, identified by `tag.fq`. Operates on the
/// active collector's table, or the host-heap descr's table when none is
/// installed; never borrows the collector.
pub fn gc_get_finalizer_queue_index(tag: FinalizerQueueTag) -> usize {
    active_finalizer_queues().get_finalizer_queue_index(tag)
}

/// Register an object with an RPython-style finalizer queue on the active GC.
pub fn gc_register_finalizer(fq_index: usize, obj: GcRef) {
    match ACTIVE_REGISTER_FINALIZER.get() {
        Some(f) => f(fq_index, obj),
        None => gc_sync::gc_op(|gc| gc.register_finalizer(fq_index, obj)),
    }
}

/// Whether a collection could still deliver a finalizer — whether
/// `deal_with_objects_with_finalizers` has anything registered to pass over, or
/// `rawrefcount` is live.
///
/// `false` means a collection can free memory but cannot run program code, so a
/// caller that collects only in order to *find* a finalizable object — teardown
/// releasing one namespace binding at a time — can skip the whole mark-and-sweep
/// and reach the same observable answer.
pub fn gc_has_pending_finalizers() -> bool {
    gc_sync::is_initialized() && gc_sync::gc_query_reentrant(|gc| gc.has_pending_death_callbacks())
}

/// How many objects are registered with a finalizer queue. A census number, for
/// diagnostics; [`gc_has_pending_finalizers`] is the question code should ask.
pub fn gc_registered_finalizer_count() -> usize {
    if gc_sync::is_initialized() {
        gc_sync::gc_query_reentrant(|gc| gc.registered_finalizer_count())
    } else {
        0
    }
}

/// Whether the object at `addr` is itself one the finalizer queue still owes a
/// delivery — `GcFlags::FINALIZER_REGISTERED` set by
/// [`GcAllocator::register_finalizer`], minus the two flags that retire it.
///
/// This is the whole of `hasuserdel` and more: `allocate_instance` registers
/// every instance of a `hasuserdel` class, and `_io`, coroutine and weakref
/// lifeline construction register objects whose *type* carries no such flag.
/// It answers only for the object itself — another object it is the last
/// referrer of stays a question only a collection can settle.
pub fn gc_object_finalizer_pending(addr: usize) -> bool {
    if !gc_owns_object(addr) {
        return false;
    }
    let hdr = unsafe { header::header_of(addr) };
    unsafe {
        (*hdr).has_flag(GcFlags::FINALIZER_REGISTERED)
            && !(*hdr).has_flag(GcFlags::FINALIZER_RUN)
            && !(*hdr).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER)
    }
}

/// incminimark.py `set_max_heap_size` — the `PYPY_GC_MAX` limit, set from
/// inside a running process instead of read once at startup.  `0` is
/// unbounded.
pub fn gc_set_max_heap_size(size: i64) {
    gc_sync::gc_op(|gc| gc.set_max_heap_size(size));
}

pub use process_ceiling::{
    arm_process_memory_ceiling, note_alloc_refused, process_memory_ceiling, try_charge, uncharge,
};

/// incminimark.py `max_heap_size_already_raised` — whether the program has
/// already been handed the one `MemoryError` a bounded heap owes it.
///
/// The next breach after that one ends the process, so this is what a caller
/// asks before running Python on its behalf: a module body, a finalizer or any
/// other nested dispatch collects on a heap that is by definition at its limit,
/// and the abort takes away whatever the caller had not finished writing.
/// Unbounded heaps never set it, so the question answers `false` unless a limit
/// is in force.
pub fn gc_max_heap_already_raised() -> bool {
    gc_sync::gc_op(|gc| gc.max_heap_already_raised())
}

/// rgc.py `move_out_of_nursery` — "Returns another object which is a copy of
/// obj; but at any point (either now or in the future) the returned object
/// might suddenly become identical to the one returned.  NOTE: Only use for
/// immutable objects!"
///
/// The argument must stay reachable across the call: filling the shadow can
/// allocate one first, and that is a collection point.
pub fn gc_move_out_of_nursery(obj: GcRef) -> GcRef {
    gc_sync::gc_op_with_root(obj, |gc, obj| GcRef(gc.move_out_of_nursery(obj.0)))
}

/// rgc.py `may_ignore_finalizer` — "says that it is valid for any finalizer
/// for 'obj' to be ignored, depending on the GC".  A hint and nothing else: the
/// object stays on its queue, and what changes is that the queue skips it when
/// it comes to deliver.
///
/// Unlike [`gc_register_finalizer`] there is no backend trampoline to consult.
/// The hint is a header bit, and a backend that keeps no such bit is a backend
/// whose queues never read one.
pub fn gc_ignore_finalizer(obj: GcRef) {
    gc_sync::gc_op(|gc| gc.ignore_finalizer(obj));
}

/// Pop one object from the active collector's finalizer death queue.
///
/// `framework.py gct_gc_fq_next_dead` is `direct_call(ll_next_dead)` where
/// `ll_next_dead` closes over its own `AddressDeque` and never enters the
/// collector.
pub fn gc_fq_next_dead(fq_index: usize) -> Option<GcRef> {
    active_finalizer_queues()
        .finalizer_deque(fq_index)
        .and_then(|deque| deque.popleft())
        .map(GcRef)
}

// ── rawrefcount ───────────────────────────────────────────────
//
// `rpython/rlib/rawrefcount.py`'s module-level surface.  Upstream's
// `ExtRegistryEntry`s (`rawrefcount.py:237-322`) lower each public name to a
// `gc_rawrefcount_*` operation, and `gctransform/framework.py:491-516` binds
// those to the collector's method of the same name — and binds them only for a
// collector that has one.  These wrappers are that binding, spelled directly.

/// `rawrefcount.py:init` — hand the collector the callback it fires when the
/// dead queue becomes non-empty.  Idempotent.
pub fn gc_rawrefcount_init(dealloc_trigger: rawrefcount::DeallocTriggerFn) {
    gc_sync::gc_op(|gc| gc.rawrefcount_init(dealloc_trigger));
}

/// Whether [`gc_rawrefcount_init`] has run on the live collector.
///
/// `as_pyobj` / `create_ref` must not invent a link before this is true:
/// `rawrefcount_create_link_pypy` asserts the table exists.
pub fn gc_rawrefcount_enabled() -> bool {
    gc_sync::is_initialized() && gc_sync::gc_query_reentrant(|gc| gc.rawrefcount_enabled())
}

/// Register the [`rawrefcount::CEdgeCensusFn`] the collector reads a block's
/// own references from.  Upstream has no counterpart; see the type.
pub fn gc_rawrefcount_set_c_edge_census(census: rawrefcount::CEdgeCensusFn) {
    gc_sync::gc_op(|gc| gc.rawrefcount_set_c_edge_census(census));
}

/// Register the [`rawrefcount::FinalizerClaimFn`] the collector claims a dying
/// block's finalizer through.  Upstream has no counterpart; see the type.
pub fn gc_rawrefcount_set_finalizer_claim(claim: rawrefcount::FinalizerClaimFn) {
    gc_sync::gc_op(|gc| gc.rawrefcount_set_finalizer_claim(claim));
}

/// Register the allocator-matched release for an exact LIGHT mirror.
pub fn gc_rawrefcount_set_light_free(free: rawrefcount::LightFreeFn) {
    gc_sync::gc_op(|gc| gc.rawrefcount_set_light_free(free));
}

/// `rawrefcount.py:create_link_pypy` — `mirror` is a mirror of `obj`, and
/// `obj` owns it: the mirror lives as long as the interpreter object does, plus
/// as long afterwards as the C side still holds a reference.
///
/// The caller must already have added [`rawrefcount::REFCNT_FROM_PYPY`] to the
/// mirror's count; that share is what this link is worth.
pub fn gc_rawrefcount_create_link_pypy(obj: GcRef, mirror: usize) {
    gc_sync::gc_op(|gc| gc.rawrefcount_create_link_pypy(obj.0, mirror));
}

/// `rawrefcount.py:create_link_pyobj` — the other direction: `mirror` owns
/// `obj`, so no trace pass roots `obj` on the mirror's behalf.
pub fn gc_rawrefcount_create_link_pyobj(obj: GcRef, mirror: usize) {
    gc_sync::gc_op(|gc| gc.rawrefcount_create_link_pyobj(obj.0, mirror));
}

/// `rawrefcount.py:mark_deallocating` — park `marker` in the link slot for the
/// duration of a mirror's deallocator, so re-entrant C code asking for the
/// interpreter object gets the sentinel rather than a freed address.
pub fn gc_rawrefcount_mark_deallocating(marker: usize, mirror: usize) {
    gc_sync::gc_op(|gc| gc.rawrefcount_mark_deallocating(marker, mirror));
}

/// `rawrefcount.py:from_obj` — `obj`'s mirror, or 0.
///
/// The collector owns this table because it is keyed by an address the
/// collector moves: a caller-side table would have to be rebuilt after every
/// minor, and could not be, because a collection must not allocate.
///
/// Without a collector there is no table, and no link can have been created
/// through one, so the answer is "no mirror" without a query.  That is the
/// state a host that has not started the runtime is in -- an object built
/// directly, outside any frame, whose type is then asked for a C slot.
pub fn gc_rawrefcount_from_obj(obj: GcRef) -> usize {
    if !gc_sync::is_initialized() {
        return 0;
    }
    gc_sync::gc_query_reentrant(|gc| gc.rawrefcount_from_obj(obj.0))
}

/// `rawrefcount.py:to_obj` — the interpreter object `mirror` links to, null
/// once the link has been cleared.
pub fn gc_rawrefcount_to_obj(mirror: usize) -> GcRef {
    GcRef(gc_sync::gc_query_reentrant(|gc| {
        gc.rawrefcount_to_obj(mirror)
    }))
}

/// `rawrefcount.py:next_dead` — pop one mirror whose linked object has died, or
/// 0 when the queue is empty.
pub fn gc_rawrefcount_next_dead() -> usize {
    gc_sync::gc_op(|gc| gc.rawrefcount_next_dead())
}

/// The next mirror whose finalizer the drain owes a call, or zero.
pub fn gc_rawrefcount_next_finalize() -> usize {
    gc_sync::gc_op(|gc| gc.rawrefcount_next_finalize())
}

/// rgc.enable / rgc.disable — toggle automatic major-collection progress
/// on the process-global GC.
pub fn gc_set_enabled(enabled: bool) {
    gc_sync::gc_op(|gc| if enabled { gc.enable() } else { gc.disable() })
}

/// rgc.isenabled — read the flag [`gc_set_enabled`] writes.
///
/// `incminimark.py` is the only reader that matters: the automatic
/// major-progress path returns early while it is clear, and an explicit
/// `gc.collect()` passes `force_enabled` to get past it.
pub fn gc_isenabled() -> bool {
    // Before `store_singleton` there is no collector to suppress, and nothing
    // could have disabled automatic collection yet.
    if !gc_sync::is_initialized() {
        return true;
    }
    gc_sync::gc_query_reentrant(|gc| gc.isenabled())
}

/// Process-global callback that performs a host-side write barrier through
/// the currently active backend GC.
pub type WriteBarrierFn = fn(obj: GcRef);
pub type WriteBarrierFromArrayFn = fn(obj: GcRef, index: usize);
pub type WritebarrierBeforeCopyFn =
    fn(source: GcRef, dest: GcRef, source_start: usize, dest_start: usize, length: usize) -> bool;

global_hook!(static ACTIVE_WRITE_BARRIER: WriteBarrierFn);
global_hook!(static ACTIVE_WRITE_BARRIER_MANAGED: WriteBarrierFn);
global_hook!(static ACTIVE_WRITE_BARRIER_BEFORE_MOVE: WriteBarrierFn);
global_hook!(static ACTIVE_WRITE_BARRIER_FROM_ARRAY: WriteBarrierFromArrayFn);
global_hook!(static ACTIVE_WRITEBARRIER_BEFORE_COPY: WritebarrierBeforeCopyFn);

/// Install the active backend's write-barrier callback. Pass `None` to clear.
pub fn set_active_write_barrier(hook: Option<WriteBarrierFn>) {
    ACTIVE_WRITE_BARRIER.set(hook);
}

/// Install the barrier entry for objects already known to be GC-managed.
pub fn set_active_write_barrier_managed(hook: Option<WriteBarrierFn>) {
    ACTIVE_WRITE_BARRIER_MANAGED.set(hook);
}

/// Install the active backend's before-move barrier. Pass `None` to clear.
pub fn set_active_write_barrier_before_move(hook: Option<WriteBarrierFn>) {
    ACTIVE_WRITE_BARRIER_BEFORE_MOVE.set(hook);
}

/// Install the active backend's `write_barrier_from_array`. Pass `None` to clear.
pub fn set_active_write_barrier_from_array(hook: Option<WriteBarrierFromArrayFn>) {
    ACTIVE_WRITE_BARRIER_FROM_ARRAY.set(hook);
}

/// Install the active backend's `writebarrier_before_copy`. Pass `None` to clear.
pub fn set_active_writebarrier_before_copy(hook: Option<WritebarrierBeforeCopyFn>) {
    ACTIVE_WRITEBARRIER_BEFORE_COPY.set(hook);
}

/// Perform a write barrier through the active backend.
///
/// Calling convention: callers must invoke this before storing a GC reference
/// into `obj`, matching [`GcAllocator::write_barrier`]. The active callback is
/// a process-global cell installed with [`set_active_write_barrier`] as a
/// [`WriteBarrierFn`]; this is a no-op when no barrier is installed.
pub fn gc_write_barrier(obj: GcRef) {
    if let Some(f) = ACTIVE_WRITE_BARRIER.get() {
        f(obj)
    }
}

/// Generalize `obj`'s cards before its items are permuted in place.
///
/// The ordinary barrier answers "a reference was stored into `obj`", which a
/// move does not do: the set of referenced objects is unchanged and only which
/// card page holds each pointer changes. A minor scans a carded array's dirty
/// pages alone, so a young pointer shifted into a clean page is not reached.
/// Calling convention: invoke this before the move, matching
/// [`MiniMarkGC::writebarrier_before_move`].
pub fn gc_write_barrier_before_move(obj: GcRef) {
    if let Some(f) = ACTIVE_WRITE_BARRIER_BEFORE_MOVE.get() {
        f(obj)
    }
}

/// Perform a write barrier without repeating the hybrid-heap ownership query.
/// Falls back to the safe barrier when a backend has no specialized entry.
pub fn gc_write_barrier_managed(obj: GcRef) {
    if let Some(f) = ACTIVE_WRITE_BARRIER_MANAGED.get() {
        f(obj)
    } else {
        gc_write_barrier(obj)
    }
}

/// incminimark.py `write_barrier_from_array(addr_array, index)`.
///
/// Host `setarrayitem_gc` uses this so a carded array records the written
/// page rather than the whole object. Upstream has one collector reached
/// through `self.gcdata.gc` (`gct_gc_writebarrier_before_copy`,
/// `transform_generic_set` in framework.py); the active hook is that
/// dispatch point here, matching [`gc_write_barrier`]. A backend's
/// `install_gc_box` stores a per-thread collector whose nursery differs
/// from the process singleton, so the hook must win while both are present.
/// `gc_sync` is the collector without a backend; with neither, the store
/// is the whole-object barrier.
pub fn gc_write_barrier_from_array(obj: GcRef, index: usize) {
    if let Some(f) = ACTIVE_WRITE_BARRIER_FROM_ARRAY.get() {
        f(obj, index);
        return;
    }
    if gc_sync::is_initialized() {
        gc_sync::gc_op(|gc| gc.write_barrier_from_array(obj, index));
        return;
    }
    gc_write_barrier(obj);
}

/// incminimark.py `writebarrier_before_copy`.
///
/// `rgc.py ll_arraycopy` calls this before `raw_memcopy`. `false` means the
/// bulk form cannot express the copy and the caller owes per-item
/// `setarrayitem_gc`. Upstream reaches `self.gcdata.gc` through
/// `gct_gc_writebarrier_before_copy` (framework.py); the active hook is
/// that dispatch point here, matching [`gc_write_barrier`]. `gc_sync` is
/// the collector without a backend; with neither, dest is remembered the
/// same way [`gc_write_barrier`] does and the copy is handled.
pub fn gc_writebarrier_before_copy(
    source: GcRef,
    dest: GcRef,
    source_start: usize,
    dest_start: usize,
    length: usize,
) -> bool {
    if let Some(f) = ACTIVE_WRITEBARRIER_BEFORE_COPY.get() {
        return f(source, dest, source_start, dest_start, length);
    }
    if gc_sync::is_initialized() {
        return gc_sync::gc_op(|gc| {
            GcAllocator::writebarrier_before_copy(
                gc,
                source,
                dest,
                source_start,
                dest_start,
                length,
            )
        });
    }
    gc_write_barrier(dest);
    true
}

// ── TEMPORARY DIAGNOSTIC: blackhole-materialized object registry ──
//
// Enabled by `MAJIT_GC_BH_PROBE`. Records every old-generation birth — the
// registration sits in `finish_alloc_in_oldgen`, so it covers the interpreter's
// own stable allocations as well as anything a backend's `bh_new*` hands out —
// so `do_collect_nursery` can check, at the end of a minor, that none of them
// still names a nursery address. `born_in` names the phase that allocated the
// block, because "born old" alone does not mean "materialized by the
// blackhole". Remove with the investigation.

thread_local! {
    /// (address, payload size, origin, allocating phase) — origin 1 = born
    /// old, 2 = promoted. The phase distinguishes a block the interpreter
    /// allocated stable from one a resume or the blackhole materialized.
    static BH_PROBE_OBJECTS: std::cell::RefCell<Vec<(usize, usize, u8, &'static str)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// One distinct `(type, offset, origin)` class of post-minor nursery reference.
pub struct BhProbeViolation {
    pub minor: usize,
    pub origin: u8,
    /// `bh_probe_phase()` at the holder's allocation — "interp", "resume",
    /// "blackhole" or "compiled".
    pub phase: &'static str,
    pub holder: usize,
    pub tid: u32,
    pub payload_size: usize,
    pub offset: usize,
    pub value: usize,
    pub forwarded: bool,
    pub type_size: usize,
    pub item_size: usize,
    pub length_offset: usize,
    pub gc_ptr_offsets: Vec<usize>,
    pub items_have_gc_ptrs: bool,
    pub custom_trace: bool,
    pub is_object: bool,
    /// Type name resolved through [`bh_probe_type_name`] at record time, while
    /// the holder is still intact; the report can run many minors later.
    pub holder_name: Option<&'static str>,
    /// Type name of the value's post-move copy. A forwarded value names a live
    /// object, so this is what actually identifies the undeclared field.
    pub value_name: Option<&'static str>,
    /// The four payload words around the violating offset, so a slot inside a
    /// Rust container (a `Vec`'s ptr/cap/len triple) is distinguishable from a
    /// genuine reference field.
    pub neighbourhood: Vec<(usize, usize)>,
    pub track_young_ptrs: bool,
    pub remembered: bool,
    pub barriered_ever: bool,
    pub traced_this_minor: bool,
    pub store_sites: u32,
    /// The object the root walk reached this holder through, `None` when the
    /// holder is itself a root. A block that carries no header of its own —
    /// an items block, a mapdict storage — is only ever traced through its
    /// owner, so the owner is what has to be on the remembered set and the
    /// only place a missing barrier can live.
    pub parent: Option<usize>,
    pub parent_tid: Option<u32>,
    pub parent_name: Option<&'static str>,
    pub parent_remembered: bool,
    pub parent_barriered_ever: bool,
    pub parent_traced_this_minor: bool,
}

thread_local! {
    static BH_PROBE_VIOLATIONS: std::cell::RefCell<Vec<BhProbeViolation>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether this `(type, offset, origin)` class was already recorded.
pub fn bh_probe_violation_seen(tid: u32, offset: usize, origin: u8) -> bool {
    BH_PROBE_VIOLATIONS
        .try_with(|v| {
            v.borrow()
                .iter()
                .any(|e| e.tid == tid && e.offset == offset && e.origin == origin)
        })
        .unwrap_or(false)
}

/// Append newly seen classes; returns the total class count.
pub fn bh_probe_record_violations(fresh: Vec<BhProbeViolation>) -> usize {
    BH_PROBE_VIOLATIONS
        .try_with(|v| {
            let mut v = v.borrow_mut();
            v.extend(fresh);
            v.len()
        })
        .unwrap_or(0)
}

pub fn bh_probe_violation_report(minor: usize) -> String {
    use std::fmt::Write as _;
    let mut out = format!("BH PROBE: post-minor nursery references, through minor #{minor}\n");
    let _ = BH_PROBE_VIOLATIONS.try_with(|v| {
        for e in v.borrow().iter() {
            let _ = writeln!(
                out,
                "  minor#{} {} born_in={} holder={:#x} tid={} name={} payload={} offset={}({}) value={:#x} \
                 forwarded={} | type: size={} item_size={} length_offset={} gc_ptr_offsets={:?} \
                 items_have_gc_ptrs={} custom_trace={} is_object={} | holder: track_young={} \
                 remembered={} barriered_ever={} traced_this_minor={} store_sites={:#b} \
                 | value_type={} around={:x?} | layout={:?} \
                 | parent={:#x} tid={} name={} remembered={} barriered_ever={} traced_this_minor={}",
                e.minor,
                if e.origin == BH_PROBE_ORIGIN_PROMOTED {
                    "promoted"
                } else {
                    "born-old"
                },
                e.phase,
                e.holder,
                e.tid,
                e.holder_name.unwrap_or("?"),
                e.payload_size,
                e.offset,
                bh_probe_field_name(e.tid, e.offset),
                e.value,
                e.forwarded,
                e.type_size,
                e.item_size,
                e.length_offset,
                e.gc_ptr_offsets,
                e.items_have_gc_ptrs,
                e.custom_trace,
                e.is_object,
                e.track_young_ptrs,
                e.remembered,
                e.barriered_ever,
                e.traced_this_minor,
                e.store_sites,
                e.value_name.unwrap_or("?"),
                e.neighbourhood,
                bh_probe_layout(e.tid),
                e.parent.unwrap_or(0),
                e.parent_tid.map(|t| t as i64).unwrap_or(-1),
                e.parent_name.unwrap_or("?"),
                e.parent_remembered,
                e.parent_barriered_ever,
                e.parent_traced_this_minor,
            );
        }
    });
    out
}

/// Type ids the post-minor scan must not read as a plain word array.
static BH_PROBE_IGNORED_TIDS: parking_lot::Mutex<Vec<u32>> = parking_lot::Mutex::new(Vec::new());

/// Exempt a type from the post-minor nursery-reference scan.
///
/// A `JitFrame`'s slots are typed by its own `jf_gcmap`, not by the type table,
/// so a slot outside that map legitimately holds a stale scratch word that can
/// fall inside the nursery range.
pub fn bh_probe_ignore_tid(type_id: u32) {
    let mut v = BH_PROBE_IGNORED_TIDS.lock();
    if !v.contains(&type_id) {
        v.push(type_id);
    }
}

pub fn bh_probe_tid_ignored(type_id: u32) -> bool {
    BH_PROBE_IGNORED_TIDS.lock().contains(&type_id)
}

/// Origin tag for a block born straight into the old generation.
pub const BH_PROBE_ORIGIN_BORN_OLD: u8 = 1;
/// Origin tag for a nursery survivor the minor promoted.
pub const BH_PROBE_ORIGIN_PROMOTED: u8 = 2;
/// Origin tag for a block born young through the non-moving rawmalloc path
/// (`external_malloc(..., alloc_young=True)`), which the next minor either
/// frees or promotes.
pub const BH_PROBE_ORIGIN_BORN_YOUNG_RAW: u8 = 3;

/// Whether the blackhole-object probe is enabled.
pub fn bh_probe_enabled() -> bool {
    // wasm32-unknown-unknown has no process environment, so the guest cannot
    // read MAJIT_GC_BH_PROBE; the compile-time feature is its gate instead.
    if cfg!(target_arch = "wasm32") {
        return cfg!(feature = "bh_probe");
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAJIT_GC_BH_PROBE").is_some())
}

/// Record an old-generation block and its payload size.
pub fn note_bh_object(addr: usize, payload_size: usize, origin: u8) {
    if addr == 0 || !bh_probe_enabled() {
        return;
    }
    let phase = bh_probe_phase();
    let _ = BH_PROBE_OBJECTS.try_with(|v| v.borrow_mut().push((addr, payload_size, origin, phase)));
}

/// Visit every recorded old-generation block.
pub fn with_bh_objects<R>(f: impl FnOnce(&[(usize, usize, u8, &'static str)]) -> R) -> Option<R> {
    BH_PROBE_OBJECTS.try_with(|v| f(&v.borrow())).ok()
}

thread_local! {
    /// Objects `remember_young_pointer` has ever put on the remembered set.
    static BH_PROBE_BARRIERED: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    /// Objects `trace_and_update_object` visited during the current minor.
    static BH_PROBE_TRACED: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

pub fn bh_probe_note_barriered(addr: usize) {
    if !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_BARRIERED.try_with(|s| s.borrow_mut().insert(addr));
}

pub fn bh_probe_note_traced(addr: usize) {
    if !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_TRACED.try_with(|s| s.borrow_mut().insert(addr));
}

pub fn bh_probe_clear_traced() {
    if !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_TRACED.try_with(|s| s.borrow_mut().clear());
}

pub fn bh_probe_was_barriered(addr: usize) -> bool {
    BH_PROBE_BARRIERED
        .try_with(|s| s.borrow().contains(&addr))
        .unwrap_or(false)
}

pub fn bh_probe_was_traced(addr: usize) -> bool {
    BH_PROBE_TRACED
        .try_with(|s| s.borrow().contains(&addr))
        .unwrap_or(false)
}

thread_local! {
    /// (object, offset) -> bitmask of instrumented store sites that wrote it.
    static BH_PROBE_STORES: std::cell::RefCell<
        std::collections::HashMap<(usize, usize), u32>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

pub fn bh_probe_note_store(obj: usize, offset: usize, site: u32) {
    if obj == 0 || !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_STORES.try_with(|m| {
        *m.borrow_mut().entry((obj, offset)).or_insert(0) |= 1u32 << site;
    });
}

pub fn bh_probe_store_sites(obj: usize, offset: usize) -> u32 {
    BH_PROBE_STORES
        .try_with(|m| m.borrow().get(&(obj, offset)).copied().unwrap_or(0))
        .unwrap_or(0)
}

/// Published nursery bounds so the probe can scan without reaching the GC.
pub static BH_PROBE_NURSERY_LO: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub static BH_PROBE_NURSERY_HI: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    static BH_PROBE_PHASE: std::cell::Cell<&'static str> = const { std::cell::Cell::new("interp") };
}

/// Diagnostic phase for `MAJIT_GC_BH_PROBE`.
///
/// `blackhole.py` `BlackholeInterpreter.run` has no phase label. The label is
/// read by `note_bh_object` and `bh_probe_scan`, and both return before that
/// read unless `bh_probe_enabled` is set. `None` means this guard did not
/// touch `BH_PROBE_PHASE`, so `Drop` must not either.
pub struct BhProbePhase(Option<&'static str>);

impl BhProbePhase {
    pub fn enter(name: &'static str) -> Self {
        // `bh_probe_enabled` is a `OnceLock`, same shape as
        // `bridge_diag_enabled`. The disabled path stops here: no
        // thread-local resolve on `BlackholeInterpreter::run`.
        if !bh_probe_enabled() {
            return BhProbePhase(None);
        }
        let prev = BH_PROBE_PHASE
            .try_with(|c| c.replace(name))
            .unwrap_or("interp");
        BhProbePhase(Some(prev))
    }
}

impl Drop for BhProbePhase {
    fn drop(&mut self) {
        let Some(prev) = self.0 else {
            return;
        };
        let _ = BH_PROBE_PHASE.try_with(|c| c.set(prev));
    }
}

pub fn bh_probe_phase() -> &'static str {
    BH_PROBE_PHASE.try_with(|c| c.get()).unwrap_or("interp")
}

/// Walk every recorded blackhole object looking for a word that names the
/// nursery, and panic naming the phase that produced it.
pub fn bh_probe_scan(reason: &str) {
    if !bh_probe_enabled() {
        return;
    }
    let lo = BH_PROBE_NURSERY_LO.load(std::sync::atomic::Ordering::Relaxed);
    let hi = BH_PROBE_NURSERY_HI.load(std::sync::atomic::Ordering::Relaxed);
    if lo == 0 {
        return;
    }
    let hit = with_bh_objects(|objects| {
        for &(addr, payload_size, origin, _phase) in objects {
            // Promoted blocks are the minor's own output; this scan is about
            // what a blackhole store just left behind in a born-old block.
            if origin != BH_PROBE_ORIGIN_BORN_OLD || (addr >= lo && addr < hi) {
                continue;
            }
            // TRACK_YOUNG_PTRS clear means the object is already on the
            // remembered set, so a young field in it is legal and will be
            // traced.  Only a flagged object naming the nursery is a lost edge.
            let hdr = (addr - crate::header::GcHeader::SIZE) as *const crate::header::GcHeader;
            if unsafe { !(*hdr).has_flag(crate::GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
                continue;
            }
            for offset in (0..payload_size).step_by(std::mem::size_of::<usize>()) {
                let word = unsafe { *((addr + offset) as *const usize) };
                if word >= lo && word < hi {
                    return Some((addr, offset, word));
                }
            }
        }
        None
    });
    if let Some(Some((addr, offset, word))) = hit {
        panic!(
            "BH SCAN: blackhole object named the nursery — reason={} phase={} \
             holder={:#x} offset={}({}) value={:#x} sites={:#b}",
            reason,
            bh_probe_phase(),
            addr,
            offset,
            bh_probe_field_name(37, offset),
            word,
            bh_probe_store_sites(addr, offset),
        );
    }
}

/// Offset -> field-name table for the probe's diagnostics, published by the
/// crate that owns the layout so the probe does not have to guess.
static BH_PROBE_FIELD_NAMES: parking_lot::Mutex<Vec<(u32, usize, &'static str)>> =
    parking_lot::Mutex::new(Vec::new());

pub fn bh_probe_set_field_names(entries: &[(u32, usize, &'static str)]) {
    BH_PROBE_FIELD_NAMES.lock().extend_from_slice(entries);
}

pub fn bh_probe_layout(type_id: u32) -> Vec<(usize, &'static str)> {
    let mut rows: Vec<(usize, &'static str)> = BH_PROBE_FIELD_NAMES
        .lock()
        .iter()
        .filter(|&&(t, _, _)| t == type_id)
        .map(|&(_, o, n)| (o, n))
        .collect();
    rows.sort_unstable();
    rows
}

pub fn bh_probe_field_name(type_id: u32, offset: usize) -> &'static str {
    BH_PROBE_FIELD_NAMES
        .lock()
        .iter()
        .find(|&&(t, o, _)| t == type_id && o == offset)
        .map(|&(_, _, n)| n)
        .unwrap_or("?")
}

/// Resolves an `rclass.OBJECT`-layout payload address to its type's name.
/// Published by the crate that owns the layouts, because the collector's
/// `TypeInfo` carries no name and a numeric type id names nothing on its own.
///
/// # Safety
/// The installed function is called only for an address the collector has
/// already established is a live object of an `is_object` type.
static BH_PROBE_TYPE_NAMER: std::sync::OnceLock<fn(usize) -> Option<&'static str>> =
    std::sync::OnceLock::new();

pub fn bh_probe_set_type_namer(f: fn(usize) -> Option<&'static str>) {
    let _ = BH_PROBE_TYPE_NAMER.set(f);
}

pub fn bh_probe_type_name(addr: usize) -> Option<&'static str> {
    BH_PROBE_TYPE_NAMER.get().and_then(|f| f(addr))
}

#[cfg(test)]
mod headerless_no_collect_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A `GcAllocator` that only implements the collecting headerless form, so
    /// the default `alloc_nursery_headerless_no_collect` has to forward to it.
    struct ForwardingGc {
        headerless_calls: usize,
    }

    impl GcAllocator for ForwardingGc {
        fn alloc_nursery(&mut self, _size: usize) -> GcRef {
            GcRef(0x1000)
        }
        fn alloc_nursery_headerless(&mut self, _size: usize) -> GcRef {
            self.headerless_calls += 1;
            GcRef(0x2000)
        }
        fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
            self.alloc_nursery(size)
        }
        fn alloc_varsize(&mut self, base: usize, item: usize, len: usize) -> GcRef {
            self.alloc_nursery(base + item * len)
        }
        fn alloc_varsize_no_collect(&mut self, base: usize, item: usize, len: usize) -> GcRef {
            self.alloc_varsize(base, item, len)
        }
        fn write_barrier(&mut self, _obj: GcRef) {}
        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {}
        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
        }
        fn collect_nursery(&mut self) {}
        fn collect_full(&mut self) {}
        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
        fn nursery_free_addr(&self) -> usize {
            0
        }
        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }
        fn nursery_top_addr(&self) -> usize {
            0
        }
        fn max_nursery_object_size(&self) -> usize {
            0
        }
    }

    #[test]
    fn no_collect_default_forwards_to_the_collecting_headerless_form() {
        let mut gc = ForwardingGc {
            headerless_calls: 0,
        };
        assert_eq!(gc.alloc_nursery_headerless_no_collect(16), GcRef(0x2000));
        assert_eq!(gc.headerless_calls, 1);
    }

    fn stub_alloc(size: usize) -> GcRef {
        GcRef(0x4000 + size)
    }

    #[test]
    fn active_hook_round_trips_and_is_null_when_absent() {
        // Uninstalled: callers must see null so they can keep their own path.
        set_active_alloc_nursery_headerless_no_collect(None);
        assert_eq!(alloc_nursery_headerless_no_collect(16), GcRef(0));

        set_active_alloc_nursery_headerless_no_collect(Some(stub_alloc));
        assert_eq!(alloc_nursery_headerless_no_collect(16), GcRef(0x4010));

        set_active_alloc_nursery_headerless_no_collect(None);
        assert_eq!(alloc_nursery_headerless_no_collect(16), GcRef(0));
    }

    static LAST_UNPINNED: AtomicUsize = AtomicUsize::new(0);

    fn stub_pin(obj: GcRef) -> bool {
        obj == GcRef(0x5010)
    }

    fn stub_unpin(obj: GcRef) {
        LAST_UNPINNED.store(obj.0, Ordering::SeqCst);
    }

    fn stub_is_pinned(obj: GcRef) -> bool {
        obj == GcRef(0x5010)
    }

    #[test]
    fn active_pin_hooks_expose_the_rgc_surface() {
        LAST_UNPINNED.store(0, Ordering::SeqCst);
        let _guard = override_gc_guard_hooks_for_test(ActiveGcGuardHooks {
            pin: Some(stub_pin),
            unpin: Some(stub_unpin),
            is_pinned: Some(stub_is_pinned),
            ..Default::default()
        });

        assert!(!pin(GcRef::NULL));
        assert!(pin(GcRef(0x5010)));
        assert!(!pin(GcRef(0x6010)));
        assert!(is_pinned(GcRef(0x5010)));
        assert!(!is_pinned(GcRef::NULL));
        unpin(GcRef(0x5010));
        assert_eq!(LAST_UNPINNED.load(Ordering::SeqCst), 0x5010);
    }

    /// incminimark.py names each `GCFLAG_*` at a position, and so does
    /// `GcFlags`.  Nothing tied the two together, so a bit corrected upstream
    /// and not here -- or the reverse -- would leave the collector reading a
    /// different bit than the one it means, with nothing noticing.
    ///
    /// `GcFlags::FLAGS` is what makes the walk complete: it carries every
    /// constant the `bitflags!` block declares, so a flag added there is
    /// compared here without anyone remembering to list it.
    #[test]
    fn every_gcflag_is_the_bit_incminimark_gives_it() {
        use bitflags::Flags as _;

        const INCMINIMARK: &str = include_str!("../../../rpython/memory/gc/incminimark.py");

        // `GCFLAG_X = first_gcflag << N`, and `GCFLAG_X = GCFLAG_Y` for the one
        // that names another flag rather than a shift.  `_GCFLAG_FIRST_UNUSED`
        // marks the end rather than a bit and does not reach either arm.
        let mut upstream: Vec<(&str, u64)> = Vec::new();
        let mut shifts = 0;
        for line in INCMINIMARK.lines() {
            let Some(rest) = line.strip_prefix("GCFLAG_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once('=') else {
                continue;
            };
            let (name, body) = (name.trim_end(), body.trim_start());
            if let Some(shift) = body.strip_prefix("first_gcflag <<") {
                let digits: String = shift
                    .trim_start()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let Ok(bit) = digits.parse::<u32>() else {
                    continue;
                };
                upstream.push((name, 1 << bit));
                shifts += 1;
            } else if let Some(alias) = body.strip_prefix("GCFLAG_") {
                let alias = alias.split_whitespace().next().unwrap_or_default();
                let (_, bit) = upstream
                    .iter()
                    .find(|(known, _)| *known == alias)
                    .unwrap_or_else(|| {
                        panic!("GCFLAG_{name} names GCFLAG_{alias}, which is declared later")
                    });
                upstream.push((name, *bit));
            }
        }

        // A parser that read nothing would agree with every flag below, so the
        // floor comes first.  incminimark declares 13 flags as a shift and one
        // as another flag's name.
        assert_eq!(
            (shifts, upstream.len()),
            (13, 14),
            "incminimark.py declares {shifts} flags as a shift and {} in all, \
             not the 13 and 14 this read",
            upstream.len()
        );

        let mut ported = 0;
        let mut ours = 0;
        for flag in GcFlags::FLAGS {
            // The prefix is the claim that incminimark declares this flag; the
            // walk holds the claim to it either way.
            let Some(name) = flag.name().strip_prefix("GCFLAG_") else {
                assert!(
                    !upstream.iter().any(|(known, _)| *known == flag.name()),
                    "incminimark.py declares GCFLAG_{}, so it is not this GC's own",
                    flag.name()
                );
                ours += 1;
                continue;
            };
            let bits = flag.value().bits();
            let (_, theirs) = upstream
                .iter()
                .find(|(known, _)| *known == name)
                .unwrap_or_else(|| panic!("incminimark.py declares no GCFLAG_{name}"));
            assert_eq!(
                bits, *theirs,
                "GCFLAG_{name} is {theirs:#x} in incminimark.py and {bits:#x} here"
            );
            ported += 1;
        }

        assert_eq!(
            ported + ours,
            GcFlags::FLAGS.len(),
            "the walk reached {ported} ported and {ours} local flags"
        );
    }

    /// The state a `collect_step` reports is incminimark's number, and the
    /// `gc` module publishes it to Python as `STATE_*` and `GC_STATES`.  This
    /// walks the same file the flags are walked in and rejects any state this
    /// GC numbers differently, or does not spell at all.
    #[test]
    fn every_gc_state_is_the_number_incminimark_gives_it() {
        const INCMINIMARK: &str = include_str!("../../../rpython/memory/gc/incminimark.py");

        // `STATE_X = N`, at column zero; the file has no other assignment
        // spelled that way.
        let mut upstream: Vec<(&str, u8)> = Vec::new();
        for line in INCMINIMARK.lines() {
            let Some(rest) = line.strip_prefix("STATE_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once('=') else {
                continue;
            };
            let Ok(value) = body.trim().parse::<u8>() else {
                continue;
            };
            upstream.push((name.trim_end(), value));
        }

        assert_eq!(
            upstream.len(),
            ALL_GC_STATES.len(),
            "incminimark.py declares {} states and this GC declares {}",
            upstream.len(),
            ALL_GC_STATES.len()
        );

        for (name, theirs) in &upstream {
            let found = ALL_GC_STATES
                .iter()
                .find(|(known, _)| known.strip_prefix("STATE_") == Some(name));
            let Some((_, ours)) = found else {
                panic!(
                    "incminimark.py declares STATE_{name} = {theirs}, and this GC has no STATE_{name}"
                );
            };
            assert_eq!(
                ours, theirs,
                "STATE_{name} is {theirs} in incminimark.py and {ours} here"
            );
        }

        // `GC_STATES` is the same four in the order the numbers give, and it
        // is what the `gc` module answers with.
        let listed = INCMINIMARK
            .lines()
            .find_map(|line| line.strip_prefix("GC_STATES = ["))
            .and_then(|body| body.split_once(']'))
            .expect("incminimark.py declares GC_STATES")
            .0;
        let listed: Vec<&str> = listed
            .split(',')
            .map(|name| name.trim().trim_matches('\''))
            .collect();
        let mut ordered = ALL_GC_STATES.to_vec();
        ordered.sort_by_key(|(_, value)| *value);
        let ordered: Vec<&str> = ordered
            .iter()
            .map(|(name, _)| name.strip_prefix("STATE_").unwrap())
            .collect();
        assert_eq!(
            listed, ordered,
            "incminimark.py's GC_STATES is {listed:?} and this GC's states are {ordered:?}"
        );
    }
}

#[cfg(test)]
mod writebarrier_before_copy_default_tests {
    use super::*;

    /// A `GcAllocator` that only records `write_barrier`, so the trait
    /// default of `writebarrier_before_copy` is the method under test.
    struct RecordingGc {
        barriered: Vec<GcRef>,
    }

    impl GcAllocator for RecordingGc {
        fn alloc_nursery(&mut self, _size: usize) -> GcRef {
            GcRef(0x1000)
        }
        fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
            self.alloc_nursery(size)
        }
        fn alloc_varsize(&mut self, base: usize, item: usize, len: usize) -> GcRef {
            self.alloc_nursery(base + item * len)
        }
        fn alloc_varsize_no_collect(&mut self, base: usize, item: usize, len: usize) -> GcRef {
            self.alloc_varsize(base, item, len)
        }
        fn write_barrier(&mut self, obj: GcRef) {
            self.barriered.push(obj);
        }
        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {}
        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
        }
        fn collect_nursery(&mut self) {}
        fn collect_full(&mut self) {}
        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
        fn nursery_free_addr(&self) -> usize {
            0
        }
        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }
        fn nursery_top_addr(&self) -> usize {
            0
        }
        fn max_nursery_object_size(&self) -> usize {
            0
        }
    }

    #[test]
    fn trait_default_writebarrier_before_copy_remembers_dest_and_returns_true() {
        let mut gc = RecordingGc {
            barriered: Vec::new(),
        };
        let source = GcRef(0x1000);
        let dest = GcRef(0x2000);
        assert!(gc.writebarrier_before_copy(source, dest, 0, 0, 4));
        assert_eq!(gc.barriered, vec![dest]);
    }
}

#[cfg(test)]
mod write_barrier_dispatch_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static FROM_ARRAY_CALLED: AtomicBool = AtomicBool::new(false);
    static FROM_ARRAY_OBJ: AtomicUsize = AtomicUsize::new(0);
    static FROM_ARRAY_INDEX: AtomicUsize = AtomicUsize::new(0);
    static BEFORE_COPY_CALLED: AtomicBool = AtomicBool::new(false);
    static BEFORE_COPY_DEST: AtomicUsize = AtomicUsize::new(0);

    fn stub_from_array(obj: GcRef, index: usize) {
        FROM_ARRAY_CALLED.store(true, Ordering::SeqCst);
        FROM_ARRAY_OBJ.store(obj.0, Ordering::SeqCst);
        FROM_ARRAY_INDEX.store(index, Ordering::SeqCst);
    }

    fn stub_before_copy(
        _source: GcRef,
        dest: GcRef,
        _source_start: usize,
        _dest_start: usize,
        _length: usize,
    ) -> bool {
        BEFORE_COPY_CALLED.store(true, Ordering::SeqCst);
        BEFORE_COPY_DEST.store(dest.0, Ordering::SeqCst);
        false
    }

    struct RestoreWriteBarrierHooks {
        from_array: Option<WriteBarrierFromArrayFn>,
        before_copy: Option<WritebarrierBeforeCopyFn>,
    }
    impl Drop for RestoreWriteBarrierHooks {
        fn drop(&mut self) {
            set_active_write_barrier_from_array(self.from_array);
            set_active_writebarrier_before_copy(self.before_copy);
        }
    }

    #[test]
    fn active_from_array_and_before_copy_hooks_win_over_gc_sync() {
        let _lock = GUARD_HOOKS_TEST_LOCK.lock();
        if !gc_sync::is_initialized() {
            gc_sync::store_singleton(Box::new(collector::MiniMarkGC::new()));
        }
        assert!(
            gc_sync::is_initialized(),
            "the singleton must be live so a reversed dispatch would skip the hook"
        );

        let _restore = RestoreWriteBarrierHooks {
            from_array: ACTIVE_WRITE_BARRIER_FROM_ARRAY.get(),
            before_copy: ACTIVE_WRITEBARRIER_BEFORE_COPY.get(),
        };
        FROM_ARRAY_CALLED.store(false, Ordering::SeqCst);
        BEFORE_COPY_CALLED.store(false, Ordering::SeqCst);
        set_active_write_barrier_from_array(Some(stub_from_array));
        set_active_writebarrier_before_copy(Some(stub_before_copy));

        let obj = GcRef(0xabc0);
        gc_write_barrier_from_array(obj, 7);
        assert!(FROM_ARRAY_CALLED.load(Ordering::SeqCst));
        assert_eq!(FROM_ARRAY_OBJ.load(Ordering::SeqCst), 0xabc0);
        assert_eq!(FROM_ARRAY_INDEX.load(Ordering::SeqCst), 7);

        let source = GcRef(0x1000);
        let dest = GcRef(0x2000);
        assert!(!gc_writebarrier_before_copy(source, dest, 1, 2, 3));
        assert!(BEFORE_COPY_CALLED.load(Ordering::SeqCst));
        assert_eq!(BEFORE_COPY_DEST.load(Ordering::SeqCst), 0x2000);
    }
}
