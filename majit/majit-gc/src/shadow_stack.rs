//! Shadow stack for GC root tracking in compiled JIT code.
//!
//! RPython reference: rpython/jit/backend/llsupport/gc.py GcRootMap_shadowstack
//!
//! Two stacks:
//! 1. GcRef shadow stack — individual GC refs (legacy, for non-jitframe roots)
//! 2. JitFrame shadow stack — jitframe pointers (RPython _call_header_shadowstack)
//!
//! Protocol for jitframe shadow stack (`_call_header_shadowstack` /
//! `_call_footer_shadowstack`):
//!   Entry: inline MOVs push [is_minor=1, jf_ptr] to root stack
//!   Per-call: push_gcmap writes jf_gcmap; pop_gcmap clears it
//!   GC: walk_jf_roots → read jf_gcmap → trace ref slots
//!   Exit: pop_jf_to(depth)   — _call_footer_shadowstack
//!
//! The jitframe shadow stack uses a per-thread flat memory array with a
//! root_stack_top pointer, matching RPython's per-thread ShadowStackPool.
//! Compiled code manipulates the current thread's root_stack_top with inline
//! load/store instructions (no function calls), exactly as in
//! `_call_header_shadowstack`.
use parking_lot::RwLock;
use std::cell::{Cell, RefCell, UnsafeCell};
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use majit_ir::GcRef;

/// shadowstack.py:281 default root_stack_depth. RPython allocates
/// `163840 * sizeofaddr` bytes for the initial shadow stack pool.
const DEFAULT_SHADOW_STACK_DEPTH: usize = 163840;

/// WORD size (bytes). RPython: arch.py WORD.
const WORD: usize = std::mem::size_of::<usize>();

// ── Heap-allocated growable jitframe shadow stack (ShadowStackPool parity) ─
//
// RPython's ShadowStackPool starts with `DEFAULT_ROOT_STACK_DEPTH`
// entries and grows via `increase_root_stack_depth` which
// `raw_malloc`s a larger buffer and copies the used portion
// (rpython/memory/gctransform/shadowstack.py). Pyre mirrors this
// with one heap-allocated `Box<[usize]>` per OS thread. Compiled JIT code
// reads/writes that thread's `root_stack_top` cell (a stable TLS address)
// and dereferences it just like RPython.
//
// Each entry is [is_minor_marker: WORD, jf_ptr: WORD] = 2*WORD bytes.
// root_stack_top points past the last entry (next free slot).
//
// assembler.py:1122-1128 _call_header_shadowstack:
//   MOV ebx, [root_stack_top_addr]    // load top pointer
//   MOV [ebx], 1                      // is_minor marker
//   MOV [ebx + WORD], ebp             // jf_ptr
//   ADD ebx, 2*WORD                   // advance
//   MOV [root_stack_top_addr], ebx    // store new top
//
// assembler.py:1130-1136 _call_footer_shadowstack:
//   SUB [root_stack_top_addr], 2*WORD  // decrement top

struct JitFrameShadowStack {
    /// Base pointer of the currently allocated backing buffer. 0 until
    /// first use. The buffer is owned by `owner`, so this address remains
    /// valid until the next grow.
    base: usize,
    /// Current top pointer. Compiled code embeds the address of this cell and
    /// mutates its inner usize directly with inline loads/stores.
    top: Cell<usize>,
    /// One-past-the-end address of the current backing buffer. Compiled code
    /// embeds this cell's address, never its value: `grow()` replaces `owner`.
    limit: Cell<usize>,
    /// Current capacity in entries (each entry is two usize words).
    capacity: usize,
    owner: Option<Box<[usize]>>,
}

impl JitFrameShadowStack {
    fn new() -> Self {
        Self {
            base: 0,
            top: Cell::new(0),
            limit: Cell::new(0),
            capacity: 0,
            owner: None,
        }
    }

    fn ensure_init(&mut self) {
        if self.base == 0 {
            self.grow(DEFAULT_SHADOW_STACK_DEPTH);
        }
    }

    /// rpython/memory/gctransform/shadowstack.py increase_root_stack_depth
    /// `increase_root_stack_depth` parity: allocate new, copy used
    /// portion, update pointers, free old. RPython also handles all live
    /// thread shadow stacks; this Rust port keeps the same shape by giving
    /// each OS thread its own backing buffer.
    fn grow(&mut self, new_capacity: usize) {
        if new_capacity <= self.capacity && self.owner.is_some() {
            return;
        }
        let used_bytes = self.top.get().saturating_sub(self.base);
        let mut new_buf: Box<[usize]> = vec![0usize; new_capacity * 2].into_boxed_slice();
        let new_ptr = new_buf.as_mut_ptr() as usize;

        if self.base != 0 && used_bytes > 0 {
            // SAFETY: `self.base` points into the previous buffer held by
            // `self.owner`; the destination is a fresh allocation. This grow
            // operation is only called from the owning thread.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.base as *const u8,
                    new_ptr as *mut u8,
                    used_bytes,
                );
            }
        }

        self.base = new_ptr;
        self.top.set(new_ptr + used_bytes);
        self.capacity = new_capacity;
        self.limit.set(new_ptr + new_capacity * 2 * WORD);
        self.owner = Some(new_buf);
    }

    fn top_addr(&self) -> usize {
        self.top.as_ptr() as usize
    }

    fn limit_addr(&self) -> usize {
        self.limit.as_ptr() as usize
    }
}

/// Callback type for tracing a libc-allocated jitframe's interior.
/// The jitframe lives in malloc memory (not nursery, not oldgen), so
/// `trace_and_update_object` can't reach it via `self.types` lookup.
/// The host crate registers a tracer that knows the JitFrame layout
/// and walks `jf_gcmap` bits to expose Ref slots.
///
/// The callback receives the jitframe payload address and a closure
/// that maps a nursery-pointing slot to its new forwarded address.
pub type LibcJitframeTracer = unsafe fn(obj_addr: usize, update: &mut dyn FnMut(*mut GcRef));

static LIBC_JF_TRACER: std::sync::OnceLock<LibcJitframeTracer> = std::sync::OnceLock::new();

/// Register the host's libc-jitframe tracer. Call once at GC init.
/// Subsequent calls are ignored (OnceLock semantics).
pub fn register_libc_jitframe_tracer(tracer: LibcJitframeTracer) {
    let _ = LIBC_JF_TRACER.set(tracer);
}

// Live libc jitframe addresses. A collector that walks the JF shadow stack
// cannot tell one of these from an unrelated foreign pointer, so the walk
// asks [`is_libc_jitframe`] before tracing interiors.
//
// The set is a flat `Vec`, not a hashed map: a thread holds one frame per
// nested compiled entry, and push / swap-remove of that list is O(1) at
// that length. `REGISTRY_USED` stays false until the first registration, so
// a process whose frames are never published (no collector is walking)
// pays one relaxed load in unregister and nothing else.
static LIBC_JF_REGISTRY_USED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static LIBC_JF_REGISTRY: OnceLock<RwLock<Vec<usize>>> = OnceLock::new();

fn libc_jf_registry() -> &'static RwLock<Vec<usize>> {
    LIBC_JF_REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

fn libc_jf_insert(set: &mut Vec<usize>, addr: usize) {
    if !set.contains(&addr) {
        set.push(addr);
    }
}

fn libc_jf_remove(set: &mut Vec<usize>, addr: usize) {
    if let Some(index) = set.iter().position(|&slot| slot == addr) {
        set.swap_remove(index);
    }
}

/// Register a libc-allocated jitframe payload address. Must be called
/// before pushing the jitframe onto the JF shadow stack.
pub fn register_libc_jitframe(addr: usize) {
    LIBC_JF_REGISTRY_USED.store(true, std::sync::atomic::Ordering::Release);
    libc_jf_insert(&mut libc_jf_registry().write(), addr);
}

/// Unregister a libc-allocated jitframe address. Call once the
/// jitframe memory is about to be freed.
pub fn unregister_libc_jitframe(addr: usize) {
    if !LIBC_JF_REGISTRY_USED.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    libc_jf_remove(&mut libc_jf_registry().write(), addr);
}

/// Check whether an address was registered as a libc-allocated jitframe.
pub fn is_libc_jitframe(addr: usize) -> bool {
    if !LIBC_JF_REGISTRY_USED.load(std::sync::atomic::Ordering::Acquire) {
        return false;
    }
    libc_jf_registry().read().contains(&addr)
}

/// Remove `addr` from [`LIBC_JF_REGISTRY`] when it is present.
///
/// `false` means the address was never registered: a host jitframe allocated
/// while no collector was installed, counted by
/// [`note_unregistered_host_jitframe`].
#[cfg(debug_assertions)]
pub fn unregister_libc_jitframe_if_present(addr: usize) -> bool {
    if !LIBC_JF_REGISTRY_USED.load(Ordering::Acquire) {
        return false;
    }
    let mut set = libc_jf_registry().write();
    if let Some(index) = set.iter().position(|&slot| slot == addr) {
        set.swap_remove(index);
        true
    } else {
        false
    }
}

/// Live host jitframes allocated while [`crate::collector_installed`] was false.
///
/// Debug-only. A release entry does not pay for the count. Those blocks are
/// not inserted into [`LIBC_JF_REGISTRY`], and an owning deadframe over one
/// is not inserted into `LIVE_DEADFRAMES`. `gc_sync::store_singleton` checks
/// the count before publishing the process collector.
#[cfg(debug_assertions)]
static UNREGISTERED_HOST_JITFRAMES: AtomicUsize = AtomicUsize::new(0);

/// Count one host jitframe allocated while no collector was installed.
#[cfg(debug_assertions)]
#[inline]
pub fn note_unregistered_host_jitframe() {
    UNREGISTERED_HOST_JITFRAMES.fetch_add(1, Ordering::Relaxed);
}

/// Drop one count taken by [`note_unregistered_host_jitframe`].
#[cfg(debug_assertions)]
#[inline]
pub fn release_unregistered_host_jitframe() {
    let prev = UNREGISTERED_HOST_JITFRAMES.fetch_sub(1, Ordering::Relaxed);
    debug_assert!(prev > 0, "unregistered host jitframe count underflow");
}

/// Fail a process-collector install that would miss live host frames.
#[cfg(debug_assertions)]
pub(crate) fn assert_no_unregistered_host_jitframes() {
    let count = UNREGISTERED_HOST_JITFRAMES.load(Ordering::Relaxed);
    if count != 0 {
        panic!(
            "installing a collector with {count} host jitframe(s) allocated while no collector was installed; those frames were not entered in LIBC_JF_REGISTRY or LIVE_DEADFRAMES and will not be traced"
        );
    }
}

/// Invoke the registered tracer if any. Returns true if tracer ran.
pub fn trace_libc_jitframe(obj_addr: usize, update: &mut dyn FnMut(*mut GcRef)) -> bool {
    if let Some(tracer) = LIBC_JF_TRACER.get() {
        unsafe { tracer(obj_addr, update) };
        true
    } else {
        false
    }
}

/// gc.py get_root_stack_top_addr()
/// Returns the ADDRESS of the current thread's root_stack_top variable
/// (not its value). Compiled code uses this to emit inline loads/stores.
pub fn get_root_stack_top_addr() -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        stack.top_addr()
    })
}

/// Address of this thread's grow-synchronised root-stack limit cell.
///
/// Compiled code loads the value for every inline push. `grow()` updates it
/// after replacing the backing buffer, while the cell address itself remains
/// stable for the lifetime of the thread-local shadow-stack object.
pub fn get_root_stack_limit_addr() -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        stack.limit_addr()
    })
}

thread_local! {
    /// Thread-local shadow stack for individual GcRef roots.  No destructor:
    /// see [`ShadowStack`]; [`SHADOW_STACK_OWNER`] frees the buffer.
    static SHADOW_STACK: ShadowStack = const { ShadowStack::new() };

    /// Registered on the first [`ShadowStack::init`]; see [`ShadowStackOwner`].
    static SHADOW_STACK_OWNER: ShadowStackOwner = const { ShadowStackOwner };

    /// Long-lived translated livevar slots whose owners are not lexical
    /// `push`/`pop_to` scopes.  RPython assigns these values fixed shadow-stack
    /// frame slots; keeping a separate slot vector preserves that ownership
    /// when Rust RAII scopes nest across a `FrameBox`.
    static OWNER_ROOTS: RefCell<Vec<Option<GcRef>>> = const { RefCell::new(Vec::new()) };

    /// Interior `OWNER_ROOTS` slots that have been released but still sit
    /// below the live tail.  Popping one is what keeps `acquire_owner_root`
    /// O(1): searching for the first `None` instead is a linear scan, and a
    /// recursion `depth` frames deep pays O(depth²) just to root its frames.
    /// Trailing slots never land here — `release_owner_root` shrinks those
    /// away so an unwound recursion leaves no walked tail behind.
    static OWNER_ROOTS_FREE: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };

    /// Thread-local flat jitframe root stack. Rust tests run multiple
    /// JIT/GC tests in parallel, so using one process-global root stack lets
    /// one GC walk another test's jitframe. RPython's root stack is per
    /// thread; mirror that here while preserving the inline root_stack_top
    /// protocol for compiled code.
    static JF_ROOT_STACK: RefCell<JitFrameShadowStack> =
        RefCell::new(JitFrameShadowStack::new());

    /// Thread-local stack of blackhole interpreter register banks.
    /// blackhole.py BlackholeInterpreter.registers_r parity: each active
    /// blackhole frame's ref register file is a GC root range. The GC must
    /// trace these slots during minor collection so nursery objects held only
    /// by a blackhole register survive across collecting calls.
    static BH_REGS_STACK: RefCell<Vec<BhRegsEntry>> = RefCell::new(Vec::with_capacity(16));

    /// Thread-local roots of every pooled blackhole interpreter, registered
    /// once when the interpreter is built (`BlackholeInterpBuilder
    /// .acquire_interp`) and dropped with it. `blackhole.py`'s interpreters
    /// are GC objects: `registers_r`, `tmpreg_r`, `exception_last_value` and
    /// `virtualizable_ptr` are traced fields for the object's whole life, not
    /// only while `run()` is on the stack.
    static BH_INTERP_ROOTS: RefCell<Vec<BhInterpEntry>> = const { RefCell::new(Vec::new()) };

    /// Thread-local stack of ref slices live only during blackhole resume
    /// construction (`resume.py blackhole_from_resumedata`).  The
    /// `virtuals_cache` and each frame's `registers_r` are filled by lazily
    /// materializing virtuals (`getvirtual_ptr` → allocator); a minor
    /// collection triggered while materializing a later virtual relocates
    /// the earlier ones, but the unrooted raw `Vec` copies are not forwarded
    /// until `run()` re-roots `registers_r` via `push_bh_regs`.  RPython
    /// traces these through the GC-managed reader/blackhole objects; pyre
    /// stores raw `i64`, so the slices are registered here for the
    /// construction window and popped when it ends.
    static RESUME_REF_ROOTS_STACK: RefCell<Vec<ResumeRefRoots>> =
        RefCell::new(Vec::with_capacity(16));
}

/// Root structures owned by one registered mutator thread.
///
/// The pointers remain valid until `unregister_mutator` runs on the owning
/// thread. Foreign access is only permitted while gc_sync has quiesced every
/// registered mutator, so none of the underlying TLS structures can move or
/// be mutably accessed by its owner during a walk.
struct MutatorEntry {
    thread_id: std::thread::ThreadId,
    shadow_stack: *const ShadowStack,
    owner_roots: *const RefCell<Vec<Option<GcRef>>>,
    jf_root_stack: *const RefCell<JitFrameShadowStack>,
    bh_regs_stack: *const RefCell<Vec<BhRegsEntry>>,
    bh_interp_roots: *const RefCell<Vec<BhInterpEntry>>,
    resume_ref_roots_stack: *const RefCell<Vec<ResumeRefRoots>>,
    extra_areas: Vec<Option<MutatorExtraArea>>,
    pruners: Vec<MutatorPruner>,
    young_reconcilers: Vec<MutatorYoungReconciler>,
}

/// Walker for one opaque root area owned by a registered mutator.
///
/// The callback runs on the collecting thread. It must derive every
/// thread-specific address from `data` and must not consult caller TLS.
pub type MutatorExtraWalkFn = unsafe fn(*const (), &mut dyn FnMut(&mut GcRef));

#[derive(Clone, Copy)]
struct MutatorExtraArea {
    walk: MutatorExtraWalkFn,
    data: *const (),
    /// Label the collector stamps on the refs this area yields.  Without it a
    /// root that fails validation reports the bare `extra_area`, which names
    /// the mechanism and not the registrar — and there are fifteen of them.
    name: &'static str,
    scoped: bool,
}

std::thread_local! {
    /// The area [`walk_all_extra_areas`] / [`walk_my_extra_areas`] is currently
    /// inside, on the collecting thread.  `extra_area` outside a walk.
    static CURRENT_EXTRA_AREA: std::cell::Cell<&'static str> =
        const { std::cell::Cell::new("extra_area") };
}

/// The registered area whose walk is running on this thread, for a collector
/// diagnostic that wants to name where a ref came from.
pub fn current_extra_area() -> &'static str {
    CURRENT_EXTRA_AREA.with(|cell| cell.get())
}

/// Restores [`CURRENT_EXTRA_AREA`] on every exit path out of
/// [`walk_one_extra_area`], an unwind included.  A plain store after the call
/// is skipped when the area's walk panics, which leaves the finished area's
/// label installed and makes the next validation failure name the wrong
/// registrar -- the opposite of what the label exists for.
pub struct ExtraAreaLabel(&'static str);

impl Drop for ExtraAreaLabel {
    fn drop(&mut self) {
        CURRENT_EXTRA_AREA.with(|cell| cell.set(self.0));
    }
}

/// Stamp [`current_extra_area`] until the guard drops.
pub fn extra_area_scope(name: &'static str) -> ExtraAreaLabel {
    ExtraAreaLabel(CURRENT_EXTRA_AREA.with(|cell| cell.replace(name)))
}

/// Run one area's walk with [`current_extra_area`] naming it.
fn walk_one_extra_area(area: &MutatorExtraArea, visitor: &mut dyn FnMut(&mut GcRef)) {
    let _label = ExtraAreaLabel(CURRENT_EXTRA_AREA.with(|cell| cell.replace(area.name)));
    // SAFETY: the callers below establish the quiescence / ownership
    // precondition; this helper only brackets the call.
    unsafe { (area.walk)(area.data, visitor) };
}

/// Pruner for one owner-keyed side table owned by a registered mutator.
///
/// Same contract as [`MutatorExtraWalkFn`]: it runs on the collecting thread
/// and must derive everything from `data`, never from caller TLS.
pub type MutatorPrunerFn = unsafe fn(*const (), &mut dyn FnMut(usize) -> Option<usize>);

#[derive(Clone, Copy)]
struct MutatorPruner {
    prune: MutatorPrunerFn,
    data: *const (),
}

/// Reconcile one mutator's young-address tables after a minor collection.
///
/// `classify` is [`MutatorPrunerFn`]'s; `in_nursery` answers whether an
/// address `classify` kept is still in the nursery (a pinned survivor), so
/// its entry stays young for the next minor collection.
pub type MutatorYoungReconcileFn =
    unsafe fn(*const (), &mut dyn FnMut(usize) -> Option<usize>, &dyn Fn(usize) -> bool);

#[derive(Clone, Copy)]
struct MutatorYoungReconciler {
    reconcile: MutatorYoungReconcileFn,
    data: *const (),
}

// The raw pointers refer to TLS owned by `thread_id`. The registry only moves
// pointer values between threads; dereferencing them requires the STW
// quiescence established by gc_sync.
unsafe impl Send for MutatorEntry {}

// `gc_sync` keeps the STW pair on OS-backed primitives because a parking_lot
// condvar inherits its process-global userspace waiter queue
// (`parking_lot_core` 0.9.12 has no atfork HashTable reset).  This registry
// is locked after fork (`after_fork_child`) and again when the child starts
// and joins a Python thread (`unregister_mutator` in `RuntimeThread::Drop`).
// A contended parking_lot park in that window is the ubuntu
// `test_2_join_in_forked_process` SIGSEGV.  `std::sync::Mutex` is the same
// choice as `GcSync::quiesce`.
struct MutatorRegistry(UnsafeCell<Mutex<Vec<MutatorEntry>>>);
// SAFETY: every reader goes through [`lock_mutator_registry`] or the
// single-threaded [`after_fork_child`] rewrite.
unsafe impl Sync for MutatorRegistry {}

static MUTATOR_REGISTRY: MutatorRegistry = MutatorRegistry(UnsafeCell::new(Mutex::new(Vec::new())));

fn lock_mutator_registry() -> MutexGuard<'static, Vec<MutatorEntry>> {
    // SAFETY: the cell is written only in [`after_fork_child`], which runs
    // before the child creates any other thread.
    unsafe { (*MUTATOR_REGISTRY.0.get()).lock() }.unwrap_or_else(PoisonError::into_inner)
}

/// Register the current thread's TLS root structures for STW root walks.
/// Unregistration is the caller's, and the pairing is armed rather than
/// returned: `pyre_interpreter::module::thread`'s `RuntimeThread` thread-local
/// calls [`unregister_mutator`] from its `Drop`, and `enter_runtime_thread`
/// arms it by touching `RUNTIME_THREAD` immediately after this call.  That
/// order is what makes it correct — this function is where the thread first
/// touches all five root structures, so their destructors are registered
/// before `RuntimeThread`'s and therefore run after it, and the registry entry
/// naming them is gone before any of them is destroyed.  A guard returned from
/// here would carry the pairing in the type system instead, but it would have
/// to be stored in a thread-local to survive the call, which is the same
/// arming with an extra step.
pub fn register_mutator() {
    let thread_id = std::thread::current().id();
    // `init` registers [`SHADOW_STACK_OWNER`]'s destructor here, before
    // `RuntimeThread`'s, so the buffer outlives this registry entry.
    let shadow_stack = SHADOW_STACK.with(|stack| {
        stack.init();
        stack as *const ShadowStack
    });
    let owner_roots = OWNER_ROOTS.with(|roots| roots as *const _);
    let jf_root_stack = JF_ROOT_STACK.with(|stack| stack as *const _);
    let bh_regs_stack = BH_REGS_STACK.with(|stack| stack as *const _);
    let bh_interp_roots = BH_INTERP_ROOTS.with(|roots| roots as *const _);
    let resume_ref_roots_stack = RESUME_REF_ROOTS_STACK.with(|stack| stack as *const _);

    let mut registry = lock_mutator_registry();
    assert!(
        !registry.iter().any(|entry| entry.thread_id == thread_id),
        "mutator thread registered twice"
    );
    registry.push(MutatorEntry {
        thread_id,
        shadow_stack,
        owner_roots,
        jf_root_stack,
        bh_regs_stack,
        bh_interp_roots,
        resume_ref_roots_stack,
        extra_areas: Vec::new(),
        pruners: Vec::new(),
        young_reconcilers: Vec::new(),
    });
}

/// Whether this thread has a [`register_mutator`] entry.
///
/// Scoped extra areas panic without one. Callers that root only when a
/// moving collector can see them use this to skip the area in tests that
/// never register.
pub fn mutator_is_registered() -> bool {
    let thread_id = std::thread::current().id();
    MUTATOR_REGISTRY
        .lock()
        .iter()
        .any(|entry| entry.thread_id == thread_id)
}

/// Append an opaque root area to the current registered mutator.
///
/// # Safety
///
/// The caller must keep `data` valid until [`unregister_mutator`] runs on this
/// thread. `walk` must derive every address it dereferences from `data`, never
/// from caller TLS, because a foreign collecting thread invokes `walk` during
/// STW.
pub unsafe fn register_mutator_extra_area(
    walk: MutatorExtraWalkFn,
    data: *const (),
    name: &'static str,
) {
    register_extra_area(walk, data, name, false);
}

fn register_extra_area(
    walk: MutatorExtraWalkFn,
    data: *const (),
    name: &'static str,
    scoped: bool,
) -> usize {
    let thread_id = std::thread::current().id();
    let mut registry = lock_mutator_registry();
    let entry = registry
        .iter_mut()
        .find(|entry| entry.thread_id == thread_id)
        .expect("register_mutator_extra_area called before register_mutator");
    let index = entry.extra_areas.len();
    entry.extra_areas.push(Some(MutatorExtraArea {
        walk,
        data,
        name,
        scoped,
    }));
    index
}

/// Scoped translated-root publication, corresponding to
/// ShadowStackFrameworkGCTransformer.push_roots/pop_roots. Unlike permanent
/// mutator areas, a frame-owned area must retire before its storage is freed.
/// Slots stay fixed so independently owned frames may retire out of order.
pub struct MutatorExtraAreaGuard {
    index: usize,
    scoped: bool,
    // An area belongs to its acquiring mutator; prevent Send/Sync.
    _owner: std::marker::PhantomData<*const ()>,
}

impl MutatorExtraAreaGuard {
    /// # Safety
    /// `data` must remain valid until this guard is dropped. The callback
    /// must obey MutatorExtraWalkFn's STW/owner-thread contract, including
    /// Rust aliasing rules. Drop the guard before unregistering the mutator.
    pub unsafe fn new(walk: MutatorExtraWalkFn, data: *const (), name: &'static str) -> Self {
        Self {
            index: register_extra_area(walk, data, name, true),
            scoped: true,
            _owner: std::marker::PhantomData,
        }
    }

    /// Mutator-lifetime extra area with RAII unregistration.
    ///
    /// [`Self::new`] is the frame-owned path and must drop before
    /// [`unregister_mutator`]. `TargetToken.virtual_state` (and the short
    /// preamble's exported state) stays reachable until the compiled-loop
    /// token is discarded, which is after `thread_die` runs the mutator
    /// destructor. Those walks are mutator roots: [`unregister_mutator`]
    /// allows them to still be live, and `Drop` is a no-op once the mutator
    /// entry is gone.
    ///
    /// # Safety
    /// Same as [`Self::new`], except `data` must remain valid until this
    /// guard is dropped *or* [`unregister_mutator`] runs, whichever is first.
    pub unsafe fn permanent(walk: MutatorExtraWalkFn, data: *const (), name: &'static str) -> Self {
        Self {
            index: register_extra_area(walk, data, name, false),
            scoped: false,
            _owner: std::marker::PhantomData,
        }
    }
}

impl Drop for MutatorExtraAreaGuard {
    fn drop(&mut self) {
        let thread_id = std::thread::current().id();
        let mut registry = lock_mutator_registry();
        let Some(entry) = registry
            .iter_mut()
            .find(|entry| entry.thread_id == thread_id)
        else {
            assert!(
                !self.scoped,
                "scoped root area outlived its registered mutator"
            );
            return;
        };
        let area = entry
            .extra_areas
            .get_mut(self.index)
            .and_then(Option::take)
            .expect("releasing an inactive scoped root area");
        assert_eq!(
            area.scoped, self.scoped,
            "releasing a root area through the wrong scope"
        );
        while entry.extra_areas.last().is_some_and(Option::is_none) {
            entry.extra_areas.pop();
        }
    }
}

/// Append an owner-keyed-table pruner to the current registered mutator.
///
/// The ephemeron half of [`register_mutator_extra_area`]: a table whose keys are
/// owner addresses and whose values a walker roots needs its dead-owner entries
/// dropped, and needs it with the same per-mutator reach the root walk has.
/// [`register_ephemeron_pruner`] cannot serve TLS-owned state — it hands the
/// classifier no way to name a thread, so a major driven by one thread would
/// leave every other thread's dead-owner entries pinned.
///
/// # Safety
///
/// Same as [`register_mutator_extra_area`]: `data` must stay valid until
/// [`unregister_mutator`] runs on this thread, and `prune` must derive every
/// address it dereferences from `data`, never from caller TLS.
pub unsafe fn register_mutator_pruner(prune: MutatorPrunerFn, data: *const ()) {
    let thread_id = std::thread::current().id();
    let mut registry = lock_mutator_registry();
    let entry = registry
        .iter_mut()
        .find(|entry| entry.thread_id == thread_id)
        .expect("register_mutator_pruner called before register_mutator");
    entry.pruners.push(MutatorPruner { prune, data });
}

/// Prune every registered mutator's owner-keyed tables during STW.
///
/// Called from the same pre-sweep point as [`prune_ephemeron_tables`] and with
/// the same classifier; see that function for why only a major prunes. Callers
/// pick between this and [`prune_my_mutator_areas`] on
/// `gc_sync::mutators_quiesced()`, exactly as the root walk picks between
/// [`walk_all_extra_areas`] and [`walk_my_extra_areas`] — so a collection's
/// prune reach always equals its own root-walk reach.
pub fn prune_all_mutator_areas(classify: &mut dyn FnMut(usize) -> Option<usize>) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "prune_all_mutator_areas reaches foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        for pruner in mutator.pruners.iter() {
            // SAFETY: gc_sync has quiesced every registered owner, and each
            // pruner's data remains valid until its MutatorEntry is removed.
            unsafe { (pruner.prune)(pruner.data, classify) };
        }
    }
}

/// Append a young-address table reconciler to the current registered mutator.
///
/// The per-mutator half of [`register_young_owner_reconciler`], as
/// [`register_mutator_pruner`] is the per-mutator half of
/// [`register_ephemeron_pruner`]: the global registration names no thread, so
/// it cannot reach a table that lives in one mutator's TLS.
///
/// # Safety
///
/// Same as [`register_mutator_pruner`].
pub unsafe fn register_mutator_young_reconciler(
    reconcile: MutatorYoungReconcileFn,
    data: *const (),
) {
    let thread_id = std::thread::current().id();
    let mut registry = lock_mutator_registry();
    let entry = registry
        .iter_mut()
        .find(|entry| entry.thread_id == thread_id)
        .expect("register_mutator_young_reconciler called before register_mutator");
    entry
        .young_reconcilers
        .push(MutatorYoungReconciler { reconcile, data });
}

/// Reconcile every registered mutator's young-address tables during STW.
///
/// Called from a minor collection at the point
/// [`reconcile_young_owner_tables`] runs, with the same classifier. The reach
/// rule is [`prune_all_mutator_areas`]'s.
pub fn reconcile_all_mutator_young(
    classify: &mut dyn FnMut(usize) -> Option<usize>,
    in_nursery: &dyn Fn(usize) -> bool,
) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "reconcile_all_mutator_young reaches foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        for reconciler in mutator.young_reconcilers.iter() {
            // SAFETY: as in `prune_all_mutator_areas`.
            unsafe { (reconciler.reconcile)(reconciler.data, classify, in_nursery) };
        }
    }
}

/// Reconcile the current mutator's young-address tables.
///
/// The single-thread collection path, mirroring [`prune_my_mutator_areas`].
pub fn reconcile_my_mutator_young(
    classify: &mut dyn FnMut(usize) -> Option<usize>,
    in_nursery: &dyn Fn(usize) -> bool,
) {
    let thread_id = std::thread::current().id();
    let registry = lock_mutator_registry();
    let Some(mutator) = registry.iter().find(|entry| entry.thread_id == thread_id) else {
        return;
    };
    for reconciler in mutator.young_reconcilers.iter() {
        // SAFETY: this is the owning thread's synchronous collection path.
        unsafe { (reconciler.reconcile)(reconciler.data, classify, in_nursery) };
    }
}

/// Prune the current mutator's owner-keyed tables.
///
/// The single-thread collection path, mirroring [`walk_my_extra_areas`]:
/// callers without a registered mutator have no per-thread tables and are a
/// no-op. A collection that only walked its own roots must only prune its own
/// tables — another mutator's owner was never marked here, so its entries
/// cannot be classified.
pub fn prune_my_mutator_areas(classify: &mut dyn FnMut(usize) -> Option<usize>) {
    let thread_id = std::thread::current().id();
    let registry = lock_mutator_registry();
    let Some(mutator) = registry.iter().find(|entry| entry.thread_id == thread_id) else {
        return;
    };
    for pruner in mutator.pruners.iter() {
        // SAFETY: this is the owning thread's synchronous collection path.
        unsafe { (pruner.prune)(pruner.data, classify) };
    }
}

/// Walk every registered mutator's opaque extra root areas during STW.
pub fn walk_all_extra_areas(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_extra_areas walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        for area in mutator.extra_areas.iter().flatten() {
            // SAFETY: gc_sync has quiesced every registered owner, and each
            // area remains valid until its MutatorEntry is removed.
            walk_one_extra_area(area, &mut visitor);
        }
    }
}

/// Walk the current mutator's opaque extra root areas.
///
/// This is the single-thread collection path; callers without a registered
/// mutator have no per-thread areas and are a no-op.
pub fn walk_my_extra_areas(mut visitor: impl FnMut(&mut GcRef)) {
    let thread_id = std::thread::current().id();
    let registry = lock_mutator_registry();
    let Some(mutator) = registry.iter().find(|entry| entry.thread_id == thread_id) else {
        return;
    };
    for area in mutator.extra_areas.iter().flatten() {
        // SAFETY: this is the owning thread's synchronous collection path.
        walk_one_extra_area(area, &mut visitor);
    }
}

/// Remove the current thread from the all-thread root registry.
///
/// Call this before gc_sync::unregister_thread so RUNNING cannot reach zero
/// while an entry whose TLS is being destroyed remains visible to the
/// collector.
pub fn unregister_mutator() {
    let thread_id = std::thread::current().id();
    let mut registry = lock_mutator_registry();
    let index = registry
        .iter()
        .position(|entry| entry.thread_id == thread_id)
        .expect("unregistering an unregistered mutator thread");
    let live_scoped: Vec<&'static str> = registry[index]
        .extra_areas
        .iter()
        .flatten()
        .filter(|area| area.scoped)
        .map(|area| area.name)
        .collect();
    assert!(
        live_scoped.is_empty(),
        "unregistering a mutator with live scoped root areas: {live_scoped:?}"
    );
    registry.swap_remove(index);
}

/// Discard root areas owned by threads which disappeared across `fork()`.
///
/// PyPy's `reinit_threads()` retains only the child thread's execution
/// context.  The shadow-stack registry is the translated equivalent of that
/// per-thread root ownership.  `shadowstack.py` `thread_after_fork` writes
/// `gcdata.thread_stacks = None` and resets `main_tid`/`active_tid` — no
/// lock.  Do not lock here either: the inherited mutex may still name a
/// vanished parent's waiter (`RPyThreadAfterFork` / `ForkMutex::reinit_after_fork`).
pub fn after_fork_child() {
    let thread_id = std::thread::current().id();
    // SAFETY: the child has one thread, so no live `MutexGuard` into the
    // old value.  `ptr::write` does not drop the inherited mutex; a vanished
    // parent thread may still "hold" it (`rgil::init_mutexes`).
    unsafe {
        let mutex = &mut *MUTATOR_REGISTRY.0.get();
        let mut entries = std::mem::take(mutex.get_mut().unwrap_or_else(PoisonError::into_inner));
        entries.retain(|entry| entry.thread_id == thread_id);
        std::ptr::write(mutex, Mutex::new(entries));
    }
}

/// The GcRef shadow stack — `gcdata.root_stack_base` / `root_stack_top`
/// (`shadowstack.py`), with the `limit` cell [`JitFrameShadowStack`] also
/// keeps so a push sees the end of the buffer without a separate depth load.
///
/// Holds no `Vec` and no `RefCell`.  A thread-local carrying drop glue pays a
/// registration-state check on every access, and a `RefCell` adds a borrow
/// flag test and two stores around each push and pop; upstream's push is a
/// store and a bump of `root_stack_top`.  The buffer is raw-allocated once at
/// `root_stack_depth` slots, as `_prepare_unused_stack` does
/// (`shadowstack.py`), and released by [`ShadowStackOwner`].
struct ShadowStack {
    base: Cell<*mut GcRef>,
    top: Cell<*mut GcRef>,
    limit: Cell<*mut GcRef>,
    /// shadowstack.py:281 `root_stack_depth`, which upstream keeps on the
    /// `ShadowStackPool` beside the stack it caps rather than in a key of its
    /// own.  Growable via `increase_root_stack_depth`; can never shrink.
    max_depth: Cell<usize>,
}

impl ShadowStack {
    const fn new() -> Self {
        ShadowStack {
            base: Cell::new(std::ptr::null_mut()),
            top: Cell::new(std::ptr::null_mut()),
            limit: Cell::new(std::ptr::null_mut()),
            max_depth: Cell::new(DEFAULT_SHADOW_STACK_DEPTH),
        }
    }

    fn layout_for(depth: usize) -> std::alloc::Layout {
        std::alloc::Layout::array::<GcRef>(depth).expect("shadow stack layout")
    }

    /// Slots in use — `root_stack_top - root_stack_base`.
    #[inline(always)]
    fn len(&self) -> usize {
        let base = self.base.get();
        if base.is_null() {
            return 0;
        }
        // SAFETY: `top` is derived from `base` within the one live buffer.
        unsafe { self.top.get().offset_from(base) as usize }
    }

    fn capacity(&self) -> usize {
        let base = self.base.get();
        if base.is_null() {
            return 0;
        }
        // SAFETY: `limit` is the one-past-the-end pointer of `base`'s buffer.
        unsafe { self.limit.get().offset_from(base) as usize }
    }

    /// `_prepare_unused_stack` on first use, then
    /// `increase_root_stack_depth`: allocate `max_depth` slots, copy the used
    /// portion over and re-derive base/top/limit.  Never shrinks.
    #[cold]
    fn init(&self) {
        let depth = self.max_depth.get();
        if depth <= self.capacity() {
            return;
        }
        // Registering the free-at-exit hook here keeps the hot key
        // destructor-free.
        let _ = SHADOW_STACK_OWNER.try_with(|_| ());
        let old_base = self.base.get();
        let used = self.len();
        // SAFETY: a non-zero array layout; zeroed slots read as null roots.
        let new_base = unsafe { std::alloc::alloc_zeroed(Self::layout_for(depth)) } as *mut GcRef;
        assert!(!new_base.is_null(), "shadow stack allocation failed");
        if !old_base.is_null() {
            // SAFETY: `used` live slots, non-overlapping fresh destination.
            unsafe { std::ptr::copy_nonoverlapping(old_base, new_base, used) };
            // SAFETY: the old buffer came from this allocator with this layout;
            // callers hold depths, not slot addresses, across a push.
            unsafe { std::alloc::dealloc(old_base as *mut u8, Self::layout_for(self.capacity())) };
        }
        self.base.set(new_base);
        // SAFETY: both offsets are inside or one past the new buffer.
        unsafe {
            self.top.set(new_base.add(used));
            self.limit.set(new_base.add(depth));
        }
    }

    /// `incr_stack(1)` with the store — returns the depth the root took.
    #[inline(always)]
    fn push(&self, gcref: GcRef) -> usize {
        let mut top = self.top.get();
        if top == self.limit.get() {
            assert!(self.base.get().is_null(), "shadow stack overflow");
            self.init();
            top = self.top.get();
        }
        // SAFETY: `top` is below `limit`, so it names a live slot.
        unsafe {
            *top = gcref;
            self.top.set(top.add(1));
        }
        self.len() - 1
    }

    /// `decr_stack` to an absolute depth; a depth at or above the current one
    /// leaves the stack alone.
    #[inline(always)]
    fn truncate(&self, depth: usize) {
        if depth < self.len() {
            // SAFETY: `depth` is below the live length.
            self.top.set(unsafe { self.base.get().add(depth) });
        }
    }

    #[inline(always)]
    fn get(&self, index: usize) -> GcRef {
        assert!(index < self.len(), "shadow stack index out of range");
        // SAFETY: `index` is below the live length.
        unsafe { *self.base.get().add(index) }
    }

    /// The live slots, for a root walk.
    ///
    /// # Safety
    /// No push may run while the returned slice is alive.
    unsafe fn slots_mut(&self) -> &mut [GcRef] {
        let base = self.base.get();
        if base.is_null() {
            return &mut [];
        }
        // SAFETY: `base..top` is the live prefix of the one buffer.
        unsafe { std::slice::from_raw_parts_mut(base, self.len()) }
    }
}

/// Frees [`SHADOW_STACK`]'s buffer when the thread exits.  It lives in its own
/// thread-local key, touched only from [`ShadowStack::init`], so the key on
/// the push/read path keeps no destructor of its own.
struct ShadowStackOwner;

impl Drop for ShadowStackOwner {
    fn drop(&mut self) {
        let _ = SHADOW_STACK.try_with(|stack| {
            let base = stack.base.get();
            if base.is_null() {
                return;
            }
            let layout = ShadowStack::layout_for(stack.capacity());
            stack.base.set(std::ptr::null_mut());
            stack.top.set(std::ptr::null_mut());
            stack.limit.set(std::ptr::null_mut());
            // SAFETY: the buffer came from this allocator with this layout and
            // the cells that named it have just been cleared.
            unsafe { std::alloc::dealloc(base as *mut u8, layout) };
        });
    }
}

// ── GcRef shadow stack (individual refs) ─────────────────────────

/// Push a GC reference onto the shadow stack.
#[inline]
pub fn push(gcref: GcRef) -> usize {
    SHADOW_STACK.with(|ss| ss.push(gcref))
}

/// Pop entries from the shadow stack back to the given depth.
///
/// `shadowstack.py decr_stack` moves `root_stack_top` and returns it;
/// the popped slots are not read back, and `pop_stack` (`:104-106`) is the
/// separate operation for callers that want the value.  Shortening in place
/// mirrors that.  Returning the popped entries instead — `Vec::split_off`,
/// which this was — makes every root-scope exit a `malloc` + `memcpy` and
/// leaves the caller a `free`, and no caller reads the result.
///
/// `depth` is asserted rather than saturated: every caller pairs this with a
/// depth it pushed, so a too-large `depth` is an unbalanced scope, and
/// silently shortening to nothing would strand roots the collector still
/// needs.  A plain `assert!` and not `debug_assert!` — this crate is
/// extracted to LLBC, where debug assertions are compiled in.
///
/// `SHADOW_STACK` has no destructor, so the key is never torn down and the
/// access cannot fail even from another thread-local's destructor; once
/// [`ShadowStackOwner`] has freed the buffer the stack reads as empty.
#[inline]
pub fn pop_to(depth: usize) {
    SHADOW_STACK.with(|ss| {
        let len = ss.len();
        assert!(
            depth <= len,
            "shadow stack pop_to({depth}) above depth {len}"
        );
        ss.truncate(depth);
    });
}

/// [`pop_to`] without the balance assertion.
///
/// Not the teardown-tolerant half of a pair: the assert is the whole
/// difference, and the truncate saturates, so a depth above the current one
/// leaves the stack alone rather than shortening it.
///
/// The callers that want it are all `Drop` impls — `FrameRoot` in both the
/// jit and the call-jit evaluator, `FrameAnchor`, and the `RootGuard`s inside
/// `gc_sync`'s `gc_op_with_root` and `gc_op_add_root` — where an assertion
/// that fails while an unwind is already running aborts the process and takes
/// the original panic's report with it.  That is a choice per site, not a rule
/// about destructors: `ExportedState::release_roots` and
/// `Trace::release_roots` are `Drop` paths that keep the assert.
#[inline]
pub fn try_pop_to(depth: usize) {
    SHADOW_STACK.with(|ss| ss.truncate(depth));
}

/// Get a GcRef at the given index.
#[inline]
pub fn get(index: usize) -> GcRef {
    SHADOW_STACK.with(|ss| ss.get(index))
}

/// Acquire a fixed translated-livevar root slot.
pub fn acquire_owner_root(root: GcRef) -> usize {
    if crate::diag_p92_enabled() && !root.is_null() {
        crate::diag_stale_gcref(root.0, "acquire_owner_root");
    }
    OWNER_ROOTS.with(|roots| {
        let mut roots = roots.borrow_mut();
        let reused = OWNER_ROOTS_FREE.with(|free| free.borrow_mut().pop());
        if let Some(index) = reused {
            debug_assert!(roots[index].is_none(), "reusing a live owner-root slot");
            roots[index] = Some(root);
            index
        } else {
            let index = roots.len();
            roots.push(Some(root));
            index
        }
    })
}

/// Release a fixed translated-livevar root slot.
pub fn release_owner_root(index: usize) {
    let _ = OWNER_ROOTS.try_with(|roots| {
        let mut roots = roots.borrow_mut();
        assert!(
            index < roots.len() && roots[index].is_some(),
            "releasing an inactive owner-root slot"
        );
        roots[index] = None;
        if index + 1 < roots.len() {
            // An interior hole: hand it to the next acquire.
            let _ = OWNER_ROOTS_FREE.try_with(|free| free.borrow_mut().push(index));
            return;
        }
        // The tail — the ordinary case, since these guards nest with the Rust
        // scopes that own them.  Shrink rather than queueing, so the walked
        // range tracks the live depth, and drop the queued indices the shrink
        // just took out of range.
        while roots.last().is_some_and(Option::is_none) {
            roots.pop();
        }
        let live = roots.len();
        let _ = OWNER_ROOTS_FREE.try_with(|free| free.borrow_mut().retain(|&i| i < live));
    });
}

/// Read a fixed owner-root slot after any collector forwarding.
///
/// The slot belongs to the current thread and must still be acquired. This is
/// the handle-based counterpart of [`OwnerRootGuard::get`] for Rust-owned
/// containers whose dependency layer cannot store the guard type directly.
pub fn get_owner_root(index: usize) -> GcRef {
    OWNER_ROOTS.with(|roots| roots.borrow()[index].expect("reading an inactive owner-root slot"))
}

/// Overwrite a live owner-root slot with a newly published value.
///
/// RPython's GC transform stores into an existing frame slot when a live
/// local is reassigned; it does not acquire a second slot. Host-side guards
/// that replace an `rbigint` handle use this so the same slot names the new
/// digit array.
pub fn set_owner_root(index: usize, root: GcRef) {
    OWNER_ROOTS.with(|roots| {
        let mut roots = roots.borrow_mut();
        assert!(
            index < roots.len() && roots[index].is_some(),
            "writing an inactive owner-root slot"
        );
        roots[index] = Some(root);
    });
}

/// RAII owner for one fixed translated-livevar root slot.
pub struct OwnerRootGuard {
    index: usize,
    /// Address of the ACQUIRING thread's `OWNER_ROOTS` cell, resolved once.
    ///
    /// A guard is read back on every access to the value it roots, so paying
    /// the thread-local resolution (`_tlv_get_addr` on macOS) each time would
    /// make re-reading the root cost more than the stale copy it replaces.
    /// The cell never moves once created — the same invariant `MutatorEntry`
    /// and [`ShadowStackSlot`] rely on — and RPython's translated code
    /// likewise resolves its root-stack base once per function
    /// (`gc_enter_roots_frame`) and then addresses slots by fixed offset.
    ///
    /// It also pins the guard to one thread: `index` names a slot in that
    /// thread's `OWNER_ROOTS`, and a bare `usize` would make the guard
    /// auto-`Send`, so dropping it elsewhere would release a foreign thread's
    /// slot (or trip the inactive-slot assert).
    roots: *const RefCell<Vec<Option<GcRef>>>,
}

impl OwnerRootGuard {
    pub fn new(root: GcRef) -> Self {
        Self {
            index: acquire_owner_root(root),
            roots: OWNER_ROOTS.with(|roots| roots as *const _),
        }
    }

    #[inline]
    pub fn get(&self) -> GcRef {
        // SAFETY: `roots` is this thread's own `OWNER_ROOTS` cell, alive for
        // as long as the guard (which cannot leave the thread), and the slot
        // stays acquired until `Drop`.
        unsafe { (*self.roots).borrow()[self.index].expect("inactive owner-root guard") }
    }

    #[inline]
    pub fn set(&self, root: GcRef) {
        // SAFETY: same cell and slot lifetime as [`Self::get`].
        unsafe {
            (*self.roots).borrow_mut()[self.index] = Some(root);
        }
    }
}

impl Drop for OwnerRootGuard {
    fn drop(&mut self) {
        release_owner_root(self.index);
    }
}

/// An opaque handle to this thread's `SHADOW_STACK` cell.
///
/// The pointer is stable for the life of the owning thread (the same
/// invariant `MutatorEntry` relies on for STW walks): the thread-local cell is
/// never reallocated once created. Resolving it once and reusing it lets a hot
/// caller re-read roots without paying the thread-local resolution
/// (`_tlv_get_addr` on macOS) on every access — the root-stack base is held in
/// a fixed location and re-read cheaply, matching the C backend's
/// once-per-function `gc_enter_roots_frame` resolution and the x86 JIT's
/// register-cached root-stack top.
#[derive(Clone, Copy)]
pub struct ShadowStackSlot(*const ShadowStack);

/// Resolve this thread's `SHADOW_STACK` cell once.
///
/// [`ShadowStackSlot::push`], [`ShadowStackSlot::get`], [`ShadowStackSlot::pop_to`]
/// and [`ShadowStackSlot::depth`] reuse the handle. [`slot_get`] is the
/// unsafe free-function form of [`ShadowStackSlot::get`].
#[inline]
pub fn shadow_stack_slot() -> ShadowStackSlot {
    ShadowStackSlot(SHADOW_STACK.with(|ss| ss as *const _))
}

impl ShadowStackSlot {
    /// `push` through this resolved handle: one store and a bump of
    /// `root_stack_top` (`ShadowStackRootWalker.push_stack`).
    #[inline]
    pub fn push(&self, gcref: GcRef) -> usize {
        // SAFETY: `self.0` is this thread's `SHADOW_STACK` cell. The
        // constructors are [`shadow_stack_slot`] and [`top`], the key has no
        // destructor, and the handle is `!Send`.
        unsafe { &*self.0 }.push(gcref)
    }

    /// `get` through this resolved handle.
    #[inline]
    pub fn get(&self, index: usize) -> GcRef {
        // SAFETY: same cell lifetime as [`Self::push`].
        unsafe { &*self.0 }.get(index)
    }

    /// `depth` through this resolved handle.
    #[inline]
    pub fn depth(&self) -> usize {
        // SAFETY: same cell lifetime as [`Self::push`].
        unsafe { &*self.0 }.len()
    }

    /// `pop_to` through this resolved handle, including the balance assert.
    #[inline]
    pub fn pop_to(&self, depth: usize) {
        // SAFETY: same cell lifetime as [`Self::push`].
        let stack = unsafe { &*self.0 };
        let len = stack.len();
        assert!(
            depth <= len,
            "shadow stack pop_to({depth}) above depth {len}"
        );
        stack.truncate(depth);
    }
}

/// Remember this thread's [`ShadowStackSlot`] in `cached` when it is empty.
///
/// `shadowstack.py` `gc_enter_roots_frame` resolves the root stack once.
/// The `SHADOW_STACK` cell is never reallocated, so later reads use the
/// stored address instead of `LocalKey::with`.
#[inline]
pub fn cache_root_stack_slot(cached: &Cell<usize>) {
    if cached.get() == 0 {
        cached.set(shadow_stack_slot().0 as usize);
    }
}

/// Slot previously stored by [`cache_root_stack_slot`].
///
/// A zero word has not been resolved yet and falls back to one resolve.
#[inline]
pub fn slot_from_cached(raw: usize) -> ShadowStackSlot {
    if raw == 0 {
        shadow_stack_slot()
    } else {
        ShadowStackSlot(raw as *const ShadowStack)
    }
}

/// The slot handle and the index of the top entry, from one thread-local
/// resolve.
///
/// The caller guarantees a non-empty stack: the top entry is a root some
/// enclosing owner pushed and has not yet popped.
#[inline]
pub fn top() -> (ShadowStackSlot, usize) {
    SHADOW_STACK.with(|ss| {
        let len = ss.len();
        assert!(len > 0, "shadow stack empty");
        (ShadowStackSlot(ss as *const _), len - 1)
    })
}

/// Get the top GcRef through one thread-local resolve.
#[inline]
pub fn top_ref() -> GcRef {
    SHADOW_STACK.with(|ss| {
        let len = ss.len();
        debug_assert!(len > 0, "shadow stack empty");
        ss.get(len - 1)
    })
}

/// Top root from a slot resolved once for this function.
///
/// `gc_enter_roots_frame` resolves the root stack once per function; later
/// reads use that pointer. `top_ref` pays `LocalKey::with` on every call.
///
/// # Safety
///
/// `slot` must come from [`shadow_stack_slot`] on this thread, and the stack
/// must be non-empty.
#[inline]
pub unsafe fn slot_top_ref(slot: ShadowStackSlot) -> GcRef {
    let ss = unsafe { &*slot.0 };
    let len = ss.len();
    debug_assert!(len > 0, "shadow stack empty");
    ss.get(len - 1)
}

/// Get a GcRef at `index` through a previously resolved [`ShadowStackSlot`].
///
/// # Safety
///
/// `slot` must have been produced by [`shadow_stack_slot`] on the current
/// thread and the thread must still be alive (its `SHADOW_STACK` not yet torn
/// down).
#[inline]
pub unsafe fn slot_get(slot: ShadowStackSlot, index: usize) -> GcRef {
    slot.get(index)
}

/// Walk all entries on the GcRef shadow stack.
///
/// Null entries are skipped here, on the walk, and nowhere else: a slot may
/// legitimately hold null, because the save side stores whatever the livevar
/// holds without testing it (`shadowcolor.py`'s `expand_one_push_roots` emits a
/// `gc_save_root` per live GCREF with no value test, and marks slots unused
/// through the translation-time `make_bitmask` rather than at runtime). This is
/// `walk_stack_root`'s `if content:` in `shadowstack.py`.
pub fn walk_roots(mut visitor: impl FnMut(&mut GcRef)) {
    SHADOW_STACK.with(|ss| {
        // SAFETY: the visitor forwards slots in place and never pushes.
        for entry in unsafe { ss.slots_mut() } {
            if !entry.is_null() {
                visitor(entry);
            }
        }
    });
    OWNER_ROOTS.with(|roots| {
        for root in roots.borrow_mut().iter_mut().flatten() {
            if !root.is_null() {
                visitor(root);
            }
        }
    });
}

/// Walk every registered mutator's GcRef shadow stack during STW.
///
/// This deliberately bypasses `RefCell::borrow_mut` on the owner-root cell: a
/// RefCell borrow is a same-thread runtime check and must not be used as the
/// synchronization mechanism for foreign TLS. gc_sync's quiescence is what
/// makes the raw dereferences and in-place forwarding sound.
pub fn walk_all_roots(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_roots walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        // SAFETY: every registered mutator is quiesced, and registry removal
        // precedes the owner's RUNNING decrement and TLS destruction.
        let ss = unsafe { (*mutator.shadow_stack).slots_mut() };
        for entry in ss.iter_mut() {
            if !entry.is_null() {
                visitor(entry);
            }
        }
        let owner_roots = unsafe { &mut *(*mutator.owner_roots).as_ptr() };
        for root in owner_roots.iter_mut().flatten() {
            if !root.is_null() {
                visitor(root);
            }
        }
    }
}

/// Current depth of the GcRef shadow stack.
///
/// Returns 0 once [`ShadowStackOwner`] has freed the buffer; callers running
/// under Drop (e.g. `ExportedState::release_roots`) observe an empty stack.
#[inline]
pub fn depth() -> usize {
    SHADOW_STACK.with(|ss| ss.len())
}

/// P92 diagnostic: name whether `slot_addr` sits in the GcRef shadow stack
/// buffer or in `OWNER_ROOTS`. Both walks share `extra_area=shadow_stack`.
pub fn describe_root_slot(slot_addr: usize) -> String {
    let ss = SHADOW_STACK.with(|ss| {
        let base = ss.base.get() as usize;
        let top = ss.top.get() as usize;
        let limit = ss.limit.get() as usize;
        let len = ss.len();
        let in_ss = base != 0 && slot_addr >= base && slot_addr < top;
        let idx = if in_ss {
            (slot_addr - base) / std::mem::size_of::<GcRef>()
        } else {
            usize::MAX
        };
        let val = if in_ss {
            // SAFETY: `idx` is inside the live prefix.
            Some(unsafe { *ss.base.get().add(idx) }.0)
        } else {
            None
        };
        format!(
            "ss_base={base:#x} ss_top={top:#x} ss_limit={limit:#x} ss_len={len} \
             ss_contains={in_ss} ss_idx={idx} ss_val={val:#x?}"
        )
    });
    // `walk_roots` already holds `OWNER_ROOTS.borrow_mut()` on this path, so
    // do not `borrow()` here. `as_ptr` is the same raw view `walk_all_roots`
    // uses under STW.
    let oroot = OWNER_ROOTS.with(|roots| {
        let roots = unsafe { &*roots.as_ptr() };
        let ptr = roots.as_ptr() as usize;
        let len = roots.len();
        let bytes = len.saturating_mul(std::mem::size_of::<Option<GcRef>>());
        let in_or = ptr != 0 && slot_addr >= ptr && slot_addr < ptr.saturating_add(bytes);
        format!("or_ptr={ptr:#x} or_len={len} or_bytes={bytes} or_contains={in_or}")
    });
    format!("{ss} {oroot}")
}

/// rpython/memory/gctransform/shadowstack.py increase_root_stack_depth
/// `increase_root_stack_depth` parity. Grows BOTH:
///   * the per-thread safety cap for the generic GcRef shadow stack
///     (`ShadowStack::max_depth`); and
///   * the jitframe shadow stack's backing buffer (reallocates via
///     the per-thread backing buffer so compiled code can push up to
///     `new_depth` jitframes without running off the end).
///
/// Never shrinks. Called from `sys.setrecursionlimit` with
/// `int(new_limit * 0.001 * 163840)` (pypy/module/sys/vm.py).
pub fn increase_root_stack_depth(new_depth: usize) {
    SHADOW_STACK.with(|ss| {
        if new_depth > ss.max_depth.get() {
            ss.max_depth.set(new_depth);
            if !ss.base.get().is_null() {
                ss.init();
            }
        }
    });
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        stack.grow(new_depth);
    });
}

/// Clear both shadow stacks.
pub fn clear() {
    SHADOW_STACK.with(|ss| ss.truncate(0));
    OWNER_ROOTS.with(|roots| roots.borrow_mut().clear());
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        // Reset root_stack_top to base of the current backing buffer.
        stack.top.set(stack.base);
    });
}

// ── JitFrame shadow stack (assembler.py:1122-1136) ───────────────
//
// Flat memory array with entries: [is_minor(usize), jf_ptr(usize)].
// root_stack_top points past the last entry. Compiled code manipulates
// root_stack_top directly with inline load/store instructions.

/// Push a jitframe GcRef onto the flat shadow stack.
///
/// assembler.py:1122-1128 _call_header_shadowstack:
///   MOV [top], 1           // is_minor marker
///   MOV [top + WORD], ebp  // jf_ptr
///   ADD top, 2*WORD        // advance
pub fn push_jf(jf_ptr: GcRef) -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let top = stack.top.get() as *mut usize;
            let depth = (top as usize - base as usize) / (2 * WORD);
            assert!(
                depth < stack.capacity,
                "jf shadow stack overflow — capacity {}, depth {}",
                stack.capacity,
                depth
            );
            // assembler.py:1125: MOV [ebx], 1
            *top = 1;
            // assembler.py:1126: MOV [ebx + WORD], ebp
            *top.add(1) = jf_ptr.0;
            // assembler.py:1127: ADD ebx, 2*WORD
            let new_top = top.add(2);
            stack.top.set(new_top as usize);
            depth
        }
    })
}

/// Read the GC-updated jf_ptr from the shadow stack at the given depth.
///
/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]
///   MOV ebp, [ecx - WORD]
pub fn peek_jf(depth: usize) -> GcRef {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *const usize;
            // entry at `depth`: base[depth*2] = is_minor, base[depth*2+1] = jf_ptr
            GcRef(*base.add(depth * 2 + 1))
        }
    })
}

/// Pop jitframe entries back to the given depth.
///
/// assembler.py:1130-1136 _call_footer_shadowstack:
///   SUB [rootstacktop], 2*WORD
pub fn pop_jf_to(depth: usize) {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let new_top = base.add(depth * 2);
            stack.top.set(new_top as usize);
        }
    });
}

/// Pop the single top jitframe entry, if any.
///
/// assembler.py:1130-1136 _call_footer_shadowstack:
///   SUB [rootstacktop], 2*WORD
/// Single-access fusion of `jf_depth()` + `pop_jf_to(depth - 1)` for
/// per-call hot paths (the wasm CA return leg pops one frame per call);
/// the two-step form pays the thread-local + RefCell round-trip twice.
pub fn pop_jf_top() {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        let top = stack.top.get();
        if top > stack.base {
            stack.top.set(top - 2 * WORD);
        }
    });
}

/// Current depth of the jitframe shadow stack.
pub fn jf_depth() -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        (stack.top.get() - stack.base) / (2 * WORD)
    })
}

/// Walk jitframe shadow stack entries as GC roots.
///
/// Each jf_ptr is exposed as `&mut GcRef`. The GC treats it like any
/// other root: if it points into the nursery, the jitframe is copied
/// to old gen and the GcRef is updated in place.
///
/// The jitframe's internal ref slots are NOT traced here — that is
/// handled by `jitframe_custom_trace` via Phase 2 (remembered set +
/// custom_trace), exactly as in RPython where `root_walker.walk_roots()`
/// copies the jitframe, and then `jitframe_trace` (custom_trace hook)
/// traces the gcmap-indicated ref slots during Phase 2.
///
/// The marker word is stepped over, not consumed. `walk_stack_root`
/// (`shadowstack.py`) treats an odd slot as a skip bitmask: on a minor
/// collection it negates an unmarked one and *returns* at an already-marked
/// one, so the walk stops instead of descending further. Porting that loop
/// here would be porting a no-op. Upstream keeps one root stack per thread, so
/// the stoppers the JIT writes sit interleaved among interpreter roots and
/// returning early skips those; this stack holds jitframe pointers only, and
/// the deep interpreter stacks (`pyre_object::gc_roots`, `SHADOW_STACK`)
/// carry no stopper to stop on. The early return becomes reachable once those
/// stacks are one stack, not before.
///
/// It would also be unsound on its own: upstream disables the optimization for
/// the first minor collection after a thread switch
/// (`can_look_at_partial_stack`, `shadowstack.py`) and whenever the
/// nursery holds objects pinned before the previous minor collection
/// (`IncrementalMiniMarkGC.collect_roots_in_nursery`'s
/// `any_pinned_object_from_earlier`). The collector now keeps that state with
/// the pin headers, but this split stack still conservatively ignores every
/// stopper until interpreter and jitframe roots share one ordered stack.
///
/// The marker is still written, by every backend, because the two-word entry
/// is the layout the merge converges on; `test_jf_flat_array_layout` is what
/// currently holds the writers to it.
pub fn walk_jf_roots(mut visitor: impl FnMut(&mut GcRef)) {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let top = stack.top.get() as *mut usize;
            let mut ptr = base;
            while ptr < top {
                // ptr[0] = is_minor marker, ptr[1] = jf_ptr
                let jf_ref = &mut *(ptr.add(1) as *mut GcRef);
                if !jf_ref.is_null() {
                    visitor(jf_ref);
                }
                ptr = ptr.add(2);
            }
        }
    });
}

/// Walk every registered mutator's jitframe shadow stack during STW.
pub fn walk_all_jf_roots(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_jf_roots walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        // SAFETY: the owning mutator is quiesced and cannot change the stack
        // or its backing allocation until the STW guard resumes it.
        let stack = unsafe { &mut *(*mutator.jf_root_stack).as_ptr() };
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let top = stack.top.get() as *mut usize;
            let mut ptr = base;
            while ptr < top {
                let jf_ref = &mut *(ptr.add(1) as *mut GcRef);
                if !jf_ref.is_null() {
                    visitor(jf_ref);
                }
                ptr = ptr.add(2);
            }
        }
    }
}

/// Read the jf_ptr of the top shadow stack entry.
///
/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]
///   MOV ebp, [ecx - WORD]
///
/// After a collecting call, the GC may have copied the jitframe. The
/// shadow stack entry has been updated. Compiled code reloads jf_ptr
/// from here to get the (possibly new) address.
pub fn jf_top_ptr() -> GcRef {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let top = stack.top.get();
            if top <= stack.base {
                return GcRef::NULL;
            }
            // top points past the last entry; jf_ptr is at top - WORD
            let jf_ptr_addr = (top - WORD) as *const usize;
            GcRef(*jf_ptr_addr)
        }
    })
}

/// Read the jf_ptr one entry below the top jitframe shadow-stack entry.
///
/// While a CALL_ASSEMBLER callee is pushed, this is its caller's frame. A
/// collecting allocation performed before the callee push can update this
/// shadow-stack slot, so compiled wasm reloads its own frame from here before
/// addressing local-0-relative frame homes.
pub fn jf_under_top_ptr() -> GcRef {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let top = stack.top.get();
            if top <= stack.base + 2 * WORD {
                return GcRef::NULL;
            }
            // Each entry is [is_minor, jf_ptr]. The caller's jf_ptr is three
            // words below top: top[-1] is this callee, top[-3] its caller.
            let jf_ptr_addr = (top - 3 * WORD) as *const usize;
            GcRef(*jf_ptr_addr)
        }
    })
}

// ── Blackhole register bank shadow stack ────────────────────────
//
// blackhole.py BlackholeInterpreter.registers_r parity:
// the ref register file is part of the GC's root set during the
// blackhole interpreter's lifetime. RPython's GC scans the
// blackhole interpreter's RPython-managed Box arrays directly;
// pyre stores raw i64 ptrs in Vec<i64> so we register the active
// register banks with the collector explicitly.

/// One active blackhole register bank.
///
/// `regs_ptr` points at the start of a `Vec<i64>` slot range; `regs_len`
/// is the number of slots (working registers AND constants — constants
/// are pre-existing old-gen pointers and pass through `copy_nursery_object`
/// untouched). `tmpreg_ptr` points at the temporary ref register that
/// holds in-flight call return values.  `exc_ptr` points at the caught
/// exception slot, which holds a reference for as long as the frame's
/// handler runs.
#[derive(Clone, Copy)]
struct BhRegsEntry {
    regs_ptr: *mut i64,
    regs_len: usize,
    tmpreg_ptr: *mut i64,
    exc_ptr: *mut i64,
    /// Owning `BlackholeInterpreter`, or null when the bank has no marker
    /// to filter against. The collector calls [`BH_LIVE_REFS`] with it
    /// before visiting, so dead Ref registers are cleared at collection
    /// time rather than at every `-live-`.
    live_ctx: *const (),
}

/// The traced fields of one pooled blackhole interpreter (`blackhole.py
/// BlackholeInterpreter`: `registers_r`, `tmpreg_r`, `exception_last_value`,
/// `virtualizable_ptr`). `regs` names the `Vec` itself, so a `setposition`
/// that regrows the bank is followed.
#[derive(Clone, Copy)]
struct BhInterpEntry {
    regs: *mut Vec<i64>,
    tmpreg_ptr: *mut i64,
    exc_ptr: *mut i64,
    vable_ptr: *mut i64,
    live_ctx: *const (),
}

/// Zeroes Ref registers the frame's last `-live-` marker does not name.
/// Installed by the embedder; a null ctx means root the whole bank.
pub type BhLiveRefsFn = unsafe fn(*const (), *mut i64, usize);

static BH_LIVE_REFS: std::sync::OnceLock<BhLiveRefsFn> = std::sync::OnceLock::new();

/// Install the collection-time Ref liveness filter. First registration wins.
pub fn register_bh_live_refs(f: BhLiveRefsFn) {
    let _ = BH_LIVE_REFS.set(f);
}

fn apply_bh_live_refs(ctx: *const (), regs: *mut i64, len: usize) {
    if ctx.is_null() {
        return;
    }
    let Some(f) = BH_LIVE_REFS.get().copied() else {
        return;
    };
    unsafe { f(ctx, regs, len) };
}

// Same thread-local discipline as `BhRegsEntry`.
unsafe impl Send for BhInterpEntry {}

// `*mut i64` is not `Send` by default. The thread-local storage means we
// only ever access entries from the thread that pushed them, so this
// `unsafe impl` is sound.
unsafe impl Send for BhRegsEntry {}

/// Push a blackhole interpreter's register bank onto the GC root stack.
///
/// The caller must ensure the slice remains valid until the matching
/// `pop_bh_regs_to(depth)` is called. Returns the previous depth so the
/// caller can restore it on exit (RPython _call_header_shadowstack /
/// _call_footer_shadowstack pattern).
///
/// # Safety
/// `regs` must remain alive and pinned until pop.
pub unsafe fn push_bh_regs(regs: &mut [i64], tmpreg: &mut i64, exc: &mut i64) -> usize {
    unsafe { push_bh_regs_with_live(regs, tmpreg, exc, std::ptr::null()) }
}

/// [`push_bh_regs`] plus the interpreter pointer collection uses to apply
/// the last `-live-` marker's Ref liveness before rooting the bank.
///
/// # Safety
/// `regs` must remain alive and pinned until pop. `live_ctx` must remain
/// valid for the same window, or be null.
pub unsafe fn push_bh_regs_with_live(
    regs: &mut [i64],
    tmpreg: &mut i64,
    exc: &mut i64,
    live_ctx: *const (),
) -> usize {
    BH_REGS_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.len();
        let max = SHADOW_STACK.with(|s| s.max_depth.get());
        assert!(depth < max, "blackhole regs stack overflow");
        ss.push(BhRegsEntry {
            regs_ptr: regs.as_mut_ptr(),
            regs_len: regs.len(),
            tmpreg_ptr: tmpreg as *mut i64,
            exc_ptr: exc as *mut i64,
            live_ctx,
        });
        depth
    })
}

/// Register a pooled blackhole interpreter's traced fields as GC roots for
/// its whole life; the counterpart of the interpreter being a GC object
/// upstream. Paired with [`unregister_bh_interp`] from the interpreter's
/// `Drop`.
///
/// # Safety
/// Every pointer must stay valid, at a fixed address, until
/// `unregister_bh_interp(regs)` runs on this thread.
pub unsafe fn register_bh_interp(
    regs: *mut Vec<i64>,
    tmpreg: *mut i64,
    exc: *mut i64,
    vable: *mut i64,
) {
    unsafe { register_bh_interp_with_live(regs, tmpreg, exc, vable, std::ptr::null()) }
}

/// [`register_bh_interp`] plus the interpreter pointer for collection-time
/// Ref liveness. `live_ctx` stays valid until `unregister_bh_interp`.
///
/// # Safety
/// Every pointer must stay valid, at a fixed address, until
/// `unregister_bh_interp(regs)` runs on this thread. `live_ctx` may be null.
pub unsafe fn register_bh_interp_with_live(
    regs: *mut Vec<i64>,
    tmpreg: *mut i64,
    exc: *mut i64,
    vable: *mut i64,
    live_ctx: *const (),
) {
    BH_INTERP_ROOTS.with(|roots| {
        roots.borrow_mut().push(BhInterpEntry {
            regs,
            tmpreg_ptr: tmpreg,
            exc_ptr: exc,
            vable_ptr: vable,
            live_ctx,
        });
    });
}

/// Drop the registration made by [`register_bh_interp`] for `regs`.
pub fn unregister_bh_interp(regs: *const Vec<i64>) {
    let _ = BH_INTERP_ROOTS.try_with(|roots| {
        let mut roots = roots.borrow_mut();
        if let Some(pos) = roots.iter().position(|e| std::ptr::eq(e.regs, regs)) {
            roots.swap_remove(pos);
        }
    });
}

/// Visit one pooled interpreter's traced fields.
///
/// # Safety
/// `entry` must have been registered by `register_bh_interp` and not yet
/// unregistered.
unsafe fn visit_bh_interp_entry(entry: &BhInterpEntry, visitor: &mut dyn FnMut(&mut GcRef)) {
    let regs = unsafe { &mut *entry.regs };
    apply_bh_live_refs(entry.live_ctx, regs.as_mut_ptr(), regs.len());
    for slot in regs.iter_mut() {
        let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
        visitor(gcref);
    }
    visitor(unsafe { &mut *(entry.tmpreg_ptr as *mut GcRef) });
    visitor(unsafe { &mut *(entry.exc_ptr as *mut GcRef) });
    visitor(unsafe { &mut *(entry.vable_ptr as *mut GcRef) });
}

/// Pop blackhole register entries back to the given depth.
pub fn pop_bh_regs_to(depth: usize) {
    let _ = BH_REGS_STACK.try_with(|ss| {
        ss.borrow_mut().truncate(depth);
    });
}

/// Walk all active blackhole register banks as GC roots.
///
/// Each i64 slot is exposed as `&mut GcRef`. Slots holding non-nursery
/// pointers (constants, old-gen, NULL) are passed through unchanged by
/// `copy_nursery_object` semantics in the visitor.
pub fn walk_bh_regs(mut visitor: impl FnMut(&mut GcRef)) {
    BH_REGS_STACK.with(|ss| {
        let ss = ss.borrow();
        for entry in ss.iter() {
            // SAFETY: the BlackholeInterpreter that pushed this entry is on
            // the call stack above us (we are inside its run() body via a
            // collecting call). The Vec<i64> backing storage is pinned for
            // the lifetime of that frame.
            apply_bh_live_refs(entry.live_ctx, entry.regs_ptr, entry.regs_len);
            let slots = unsafe { std::slice::from_raw_parts_mut(entry.regs_ptr, entry.regs_len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
            // tmpreg_r holds in-flight call return values between
            // `let result = call_int_function(...)` and the subsequent
            // `self.registers_r[dst] = result;` store. RPython's blackhole
            // interpreter holds this in an RPython-managed slot so it is
            // automatically a root.
            let tmp = unsafe { &mut *(entry.tmpreg_ptr as *mut GcRef) };
            visitor(tmp);
            // `exception_last_value` is the same kind of slot: `route_to_catch`
            // stores the caught exception there and jumps to the handler
            // (`blackhole.py`), and the handler recovers it much later
            // through `last_exc_value` / `last_exception` — which dereferences
            // it via `bh_classof`. Everything the handler runs in between can
            // allocate, so the reference has to survive a minor collection.
            // Upstream needs no root here for the same reason it needs none for
            // tmpreg: the field lives on a GC-managed interpreter object.
            let exc = unsafe { &mut *(entry.exc_ptr as *mut GcRef) };
            visitor(exc);
        }
    });
    BH_INTERP_ROOTS.with(|roots| {
        for entry in roots.borrow().iter() {
            // SAFETY: registered interpreters unregister from their `Drop`.
            unsafe { visit_bh_interp_entry(entry, &mut visitor) };
        }
    });
}

/// Walk every registered mutator's blackhole register roots during STW.
pub fn walk_all_bh_regs(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_bh_regs walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        // SAFETY: all owners are quiesced, and each registered slice remains
        // pinned until its owning blackhole frame pops the entry after resume.
        let entries = unsafe { &*(*mutator.bh_regs_stack).as_ptr() };
        for entry in entries.iter() {
            apply_bh_live_refs(entry.live_ctx, entry.regs_ptr, entry.regs_len);
            let slots = unsafe { std::slice::from_raw_parts_mut(entry.regs_ptr, entry.regs_len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
            let tmp = unsafe { &mut *(entry.tmpreg_ptr as *mut GcRef) };
            visitor(tmp);
            let exc = unsafe { &mut *(entry.exc_ptr as *mut GcRef) };
            visitor(exc);
        }
        let interps = unsafe { &*(*mutator.bh_interp_roots).as_ptr() };
        for entry in interps.iter() {
            // SAFETY: as above; a registered interpreter unregisters from
            // its `Drop`, which runs on the owning (quiesced) thread.
            unsafe { visit_bh_interp_entry(entry, &mut visitor) };
        }
    }
}

/// Current depth of the blackhole register bank stack.
pub fn bh_regs_depth() -> usize {
    BH_REGS_STACK.with(|ss| ss.borrow().len())
}

/// Current depth of the resume-construction ref-slice root stack.
pub fn resume_ref_roots_depth() -> usize {
    RESUME_REF_ROOTS_STACK.with(|ss| ss.borrow().len())
}

/// Whether a ref slice starting at `ptr` is currently registered as a
/// resume-construction root on this thread.
///
/// Exists so the paths that materialize virtuals can assert their own
/// precondition — the cache slice they write into has to be forwardable in
/// place by a collection their allocations trigger — instead of arguing it.
pub fn resume_ref_slice_registered(ptr: *const i64) -> bool {
    RESUME_REF_ROOTS_STACK.with(|ss| {
        ss.borrow()
            .iter()
            .any(|entry| entry.ptr == ptr.cast_mut().cast())
    })
}

struct ResumeRefRoots {
    ptr: *mut u8,
    len: usize,
    stride: usize,
}

impl ResumeRefRoots {
    unsafe fn walk(&self, visitor: &mut impl FnMut(&mut GcRef)) {
        for index in 0..self.len {
            visitor(unsafe { &mut *self.ptr.add(index * self.stride).cast::<GcRef>() });
        }
    }
}

/// Register a pointer-width field, as opposed to an i64 JIT register bank.
/// RPython's `ResumeDataDirectReader` keeps these fields on GC-traced objects;
/// stationary Rust virtualizables expose their actual GCREF slots instead.
///
/// # Safety
/// The slot must remain alive and stationary until `pop_resume_ref_roots_to`.
#[track_caller]
pub unsafe fn push_resume_ref_root(slot: &mut GcRef) {
    assert!(
        !crate::gc_is_nursery_object(slot as *mut GcRef as usize),
        "GC BUG: resume root field points inside the movable nursery",
    );
    if crate::gc_nursery_poison_enabled() {
        assert_ne!(
            slot.0,
            (usize::MAX / 0xff) * 0xaa,
            "GC BUG: poisoned resume root field"
        );
    }
    RESUME_REF_ROOTS_STACK.with(|ss| {
        ss.borrow_mut().push(ResumeRefRoots {
            ptr: (slot as *mut GcRef).cast(),
            len: 1,
            stride: std::mem::size_of::<GcRef>(),
        });
    });
}

/// Register a ref slice as a GC root for the blackhole resume
/// construction window (`resume.py blackhole_from_resumedata`).
///
/// # Safety
/// `slice` must remain alive and at a fixed address until the matching
/// `pop_resume_ref_roots_to` runs.  Both the `virtuals_cache` and a
/// blackhole frame's `registers_r` are stable after allocation
/// (`prepare_virtuals` / `setposition` size them once; later fills only
/// index), so the captured pointer stays valid for the whole window.
#[track_caller]
pub unsafe fn push_resume_ref_roots(slice: &mut [i64]) {
    if !slice.is_empty() && crate::gc_is_nursery_object(slice.as_ptr() as usize) {
        panic!(
            "GC BUG: resume root slice points inside the movable nursery: ptr={:#x} len={} caller={}",
            slice.as_ptr() as usize,
            slice.len(),
            std::panic::Location::caller(),
        );
    }
    if crate::gc_nursery_poison_enabled() {
        const NURSERY_POISON_WORD: i64 = ((usize::MAX / 0xff) * 0xaa) as i64;
        if let Some(index) = slice.iter().position(|&value| value == NURSERY_POISON_WORD) {
            panic!(
                "GC BUG: resume root slice contains nursery poison before registration: index={} len={} caller={}",
                index,
                slice.len(),
                std::panic::Location::caller(),
            );
        }
    }
    RESUME_REF_ROOTS_STACK.with(|ss| {
        ss.borrow_mut().push(ResumeRefRoots {
            ptr: slice.as_mut_ptr().cast(),
            len: slice.len(),
            stride: std::mem::size_of::<i64>(),
        });
    });
}

/// Pop resume-construction ref-slice roots back to the given depth.
pub fn pop_resume_ref_roots_to(depth: usize) {
    let _ = RESUME_REF_ROOTS_STACK.try_with(|ss| {
        ss.borrow_mut().truncate(depth);
    });
}

/// Walk all registered resume-construction ref slices as GC roots.
///
/// Each `i64` slot is exposed as `&mut GcRef`; non-nursery values
/// (unmaterialized `0` cache slots, old-gen pointers) pass through the
/// visitor's `copy_nursery_object` guard unchanged.
pub fn walk_resume_ref_roots(mut visitor: impl FnMut(&mut GcRef)) {
    RESUME_REF_ROOTS_STACK.with(|ss| {
        for entry in ss.borrow().iter() {
            unsafe { entry.walk(&mut visitor) };
        }
    });
}

/// Visit the current context's JIT stack roots and the backend's live deadframes.
///
/// `shadowstack.py ShadowStackRootWalker.walk_stack_roots` owns this boundary
/// upstream; `blackhole.py BlackholeInterpreter` and `resume.py
/// ResumeDataDirectReader` are traced GC objects there. Here their raw register
/// banks and libc jitframes need explicit walks, which belong to majit rather
/// than to each embedding interpreter's collector.
///
/// The visitor must filter references by its heap's ownership and trace the
/// interiors of its own managed objects. Libc jitframe interiors are exposed
/// here through the backend's registered tracer, including the post-epilogue
/// DEADFRAME lifetime (`llmodel.py AbstractLLCPU.get_ref_value`). Static roots,
/// extra areas and embedding-runtime root hooks are separate root categories
/// (`framework.py BaseRootWalker.walk_roots`) and are not called by this API.
/// Stack sources are current-context only; the backend deadframe registry may
/// be process-wide. As with the collector's root phase, callers must exclude
/// concurrent mutation of visited frames. Multi-mutator collectors must also
/// walk the other mutators' stacks through the `walk_all_*` APIs.
pub fn walk_jit_roots(mut visitor: impl FnMut(&mut GcRef)) {
    walk_roots(&mut visitor);
    walk_jf_roots(|gcref| {
        visitor(gcref);
        if !gcref.is_null() && is_libc_jitframe(gcref.0) {
            trace_libc_jitframe(gcref.0, &mut |slot| unsafe { visitor(&mut *slot) });
        }
    });
    crate::walk_active_live_deadframes(&mut |addr| {
        trace_libc_jitframe(addr, &mut |slot| unsafe { visitor(&mut *slot) });
    });
    walk_bh_regs(&mut visitor);
    walk_resume_ref_roots(&mut visitor);
}

/// Walk every registered mutator's resume-construction roots during STW.
pub fn walk_all_resume_ref_roots(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_resume_ref_roots walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = lock_mutator_registry();
    for mutator in registry.iter() {
        // SAFETY: the owner is quiesced and the registered slices stay pinned
        // for the complete resume-construction window.
        let entries = unsafe { &*(*mutator.resume_ref_roots_stack).as_ptr() };
        for entry in entries.iter() {
            unsafe { entry.walk(&mut visitor) };
        }
    }
}

// ── Extra root walkers registered by the embedder ───────────────
//
// rpython/memory/gctransform/framework.py `root_walker.walk_roots` parity:
// RPython registers GcRootMap sources (shadow stack, thread-local refs,
// jitframes, stack roots) once during setup; at collection time, the GC
// iterates all registered sources.
//
// pyre's interpreter holds live GC refs in `PyFrame.locals_cells_stack_w`
// and must register a walker for that storage from outside majit-gc
// (a crate-level upward dependency would cycle). The hook lets any
// embedder (pyre-interpreter, pyre-jit) plug a callback that the
// collector will call during `Phase 1e` of `do_collect_nursery`.

/// Signature expected by `register_extra_root_walker`.
///
/// The registered function receives an opaque visitor that must be
/// invoked once per GC root slot. Each slot is exposed as
/// `&mut GcRef`; writing back through the reference forwards the
/// root if the GC moved the referenced object.
pub type ExtraRootWalkerFn = fn(&mut dyn FnMut(&mut GcRef));

/// Which collection is currently driving the extra-root walk.
///
/// incminimark distinguishes the two: a minor collection scans an
/// old/prebuilt object only when the write barrier recorded a store into it
/// (`old_objects_pointing_to_young`, incminimark.py:339-344), while a major
/// collection always traces `prebuilt_root_objects` (incminimark.py:355).
/// Walkers that mirror prebuilt-object scanning read this to apply the same
/// minor-skip; the default is [`ExtraRootWalkKind::Major`] (scan everything)
/// so an unset call site stays conservative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExtraRootWalkKind {
    Minor,
    Major,
}

thread_local! {
    static EXTRA_ROOT_WALK_KIND: std::cell::Cell<ExtraRootWalkKind> =
        const { std::cell::Cell::new(ExtraRootWalkKind::Major) };
}

/// Set by the collector around its root-walk phases; see
/// [`ExtraRootWalkKind`].
pub fn set_extra_root_walk_kind(kind: ExtraRootWalkKind) {
    EXTRA_ROOT_WALK_KIND.with(|k| k.set(kind));
}

/// The collection kind driving the current extra-root walk.
pub fn extra_root_walk_kind() -> ExtraRootWalkKind {
    EXTRA_ROOT_WALK_KIND.with(|k| k.get())
}

// One registration per *kind* of root storage, as framework.py registers
// GcRootMap sources: the interpreter's process-global off-GC slots
// (`walk_interpreter_global_roots`), the exceptions parked outside GC
// discipline (`walk_parked_exception_roots`), the immortal process-global
// stores (`walk_immortal_store_roots`), and the per-loop gc_table walker
// registered from gcreftracer.rs — four in a native build, three on wasm and
// under `sandbox`, where the faulthandler source is compiled out.
//
// Anything that enumerates a population belongs inside one of those, not in a
// slot of its own; the cap is what stops the array from drifting back into a
// per-defect callback registry, so raising it is the wrong response to running
// out of room.
const MAX_EXTRA_ROOT_WALKERS: usize = 8;

/// Registered root walkers. Each slot is either `None` or a function
/// pointer plus the label the debug walk prints. We cap the count to keep
/// the set stack-allocatable and avoid dynamic allocation in the GC hot path.
/// Collection walks the function and ignores the label.
static EXTRA_ROOT_WALKERS: parking_lot::RwLock<
    [Option<(ExtraRootWalkerFn, &'static str)>; MAX_EXTRA_ROOT_WALKERS],
> = parking_lot::RwLock::new([None; MAX_EXTRA_ROOT_WALKERS]);

/// Register an additional root walker.
///
/// Called at process start (or module init) by the embedder once per
/// root source. Duplicate registrations are tolerated — the walker is
/// only appended if not already present. `label` is debug-only: the
/// collector's root walk does not read it.
pub fn register_extra_root_walker(walker: ExtraRootWalkerFn, label: &'static str) {
    let mut guard = EXTRA_ROOT_WALKERS.write();
    for slot in guard.iter_mut() {
        match slot {
            Some((existing, _)) if std::ptr::fn_addr_eq(*existing, walker) => return,
            None => {
                *slot = Some((walker, label));
                return;
            }
            _ => {}
        }
    }
    panic!(
        "register_extra_root_walker: capacity exceeded ({} walkers already registered)",
        MAX_EXTRA_ROOT_WALKERS
    );
}

/// Walkers that `rescan_major_nonstack_roots_and_drain` must repeat and that
/// `walk_extra_roots` must not. Per-mutator `PyFrameRootArea` already walks
/// the collecting thread's TLS exception cells on the first pass; repeating
/// those three cells here is what incminimark's second `collect_nonstack_roots`
/// does for values that can appear after the initial snapshot. Putting them
/// in [`EXTRA_ROOT_WALKERS`] would double-walk them on every nursery/major
/// seed.
static RESCAN_ROOT_WALKERS: parking_lot::RwLock<
    [Option<ExtraRootWalkerFn>; MAX_EXTRA_ROOT_WALKERS],
> = parking_lot::RwLock::new([None; MAX_EXTRA_ROOT_WALKERS]);

/// Register a walker that only the mid-major non-stack rescan invokes.
pub fn register_rescan_root_walker(walker: ExtraRootWalkerFn) {
    let mut guard = RESCAN_ROOT_WALKERS.write();
    for slot in guard.iter_mut() {
        match slot {
            Some(existing) if std::ptr::fn_addr_eq(*existing, walker) => return,
            None => {
                *slot = Some(walker);
                return;
            }
            _ => {}
        }
    }
    panic!(
        "register_rescan_root_walker: capacity exceeded ({} walkers already registered)",
        MAX_EXTRA_ROOT_WALKERS
    );
}

/// Invoke every rescan-only walker. Called from
/// `MiniMarkGC::rescan_major_nonstack_roots_and_drain`.
pub fn walk_rescan_roots(mut visitor: impl FnMut(&mut GcRef)) {
    let walkers = {
        let guard = RESCAN_ROOT_WALKERS.read();
        *guard
    };
    for walker in walkers.iter().flatten() {
        walker(&mut visitor);
    }
}

/// Invoke every registered extra root walker with the given visitor.
///
/// Called by `MiniMarkGC::do_collect_nursery` (Phase 1e). Each walker's
/// label is stamped on [`current_extra_area`] so a root that fails
/// validation names the registrar, not the bare default.
pub fn walk_extra_roots(mut visitor: impl FnMut(&mut GcRef)) {
    // Snapshot the walker list under a read guard so a walker that
    // triggers further allocation (and recursively a collection) does
    // not observe the lock held in write mode.
    let walkers = {
        let guard = EXTRA_ROOT_WALKERS.read();
        *guard
    };
    for (walker, label) in walkers.iter().flatten() {
        let _label = extra_area_scope(*label);
        walker(&mut visitor);
    }
}

/// [`walk_extra_roots`] that also names the walker.
///
/// `enumerate_labeled_root_walker_values` is the reader. Collection keeps
/// using [`walk_extra_roots`].
pub fn walk_extra_roots_labeled(mut visitor: impl FnMut(&mut GcRef, &'static str)) {
    let walkers = {
        let guard = EXTRA_ROOT_WALKERS.read();
        *guard
    };
    for (walker, label) in walkers.iter().flatten() {
        let label = *label;
        walker(&mut |gcref| visitor(gcref, label));
    }
}

// ── Ephemeron side tables ───────────────────────────────────────
//
// An embedder side table keyed by owner address holds its value as a root, so
// the value outlives the owner and the table only grows.  Upstream has no such
// table — `typedef.py`'s generated subclass carries the field on the object,
// which dies with it — so the entries need the ephemeron semantics that
// layout gives for free: the value is kept only while the key is alive.

/// Signature expected by [`register_ephemeron_pruner`].
///
/// The registered function walks its own table and asks the supplied
/// classifier about each owner address: `Some(addr)` means the owner survived
/// and now lives at `addr`, `None` means it died and the entry must be
/// dropped.
pub type EphemeronPrunerFn = fn(&mut dyn FnMut(usize) -> Option<usize>);
/// A companion marker for values whose owners survived major marking.
/// Implementations append values to `roots`; the collector marks them and
/// repeats to a fixed point before finalizer ordering and pruning.
pub type EphemeronMarkerFn = fn(&mut dyn FnMut(usize) -> Option<usize>, &mut Vec<GcRef>);

const MAX_EPHEMERON_PRUNERS: usize = 4;

static EPHEMERON_PRUNERS: parking_lot::RwLock<[Option<EphemeronPrunerFn>; MAX_EPHEMERON_PRUNERS]> =
    parking_lot::RwLock::new([None; MAX_EPHEMERON_PRUNERS]);
static EPHEMERON_MARKERS: parking_lot::RwLock<[Option<EphemeronMarkerFn>; MAX_EPHEMERON_PRUNERS]> =
    parking_lot::RwLock::new([None; MAX_EPHEMERON_PRUNERS]);

/// Register a side table whose entries must be dropped when their owner dies.
///
/// Duplicate registrations are tolerated — the pruner is only appended if not
/// already present.
pub fn register_ephemeron_pruner(pruner: EphemeronPrunerFn) {
    let mut guard = EPHEMERON_PRUNERS.write();
    for slot in guard.iter_mut() {
        match slot {
            Some(existing) if std::ptr::fn_addr_eq(*existing, pruner) => return,
            None => {
                *slot = Some(pruner);
                return;
            }
            _ => {}
        }
    }
    panic!(
        "register_ephemeron_pruner: capacity exceeded ({MAX_EPHEMERON_PRUNERS} pruners already \
         registered)"
    );
}

pub fn register_ephemeron_marker(marker: EphemeronMarkerFn) {
    let mut guard = EPHEMERON_MARKERS.write();
    for slot in guard.iter_mut() {
        match slot {
            Some(existing) if std::ptr::fn_addr_eq(*existing, marker) => return,
            None => {
                *slot = Some(marker);
                return;
            }
            _ => {}
        }
    }
    panic!(
        "register_ephemeron_marker: capacity exceeded ({MAX_EPHEMERON_PRUNERS} markers already \
         registered)"
    );
}

pub fn mark_ephemeron_tables(classify: &mut dyn FnMut(usize) -> Option<usize>) -> Vec<GcRef> {
    let markers = {
        let guard = EPHEMERON_MARKERS.read();
        *guard
    };
    let mut roots = Vec::new();
    for marker in markers.iter().flatten() {
        marker(classify, &mut roots);
    }
    roots
}

/// Invoke every registered pruner with a classifier that answers whether an
/// owner survived this collection.
///
/// Called by `MiniMarkGC::finish_incremental_marking` once marking has settled
/// and before the sweep, where VISITED still tells a survivor from a dying
/// object.  Only a major collection prunes: it already costs O(live heap), so
/// an O(table) pass adds no asymptotic term, whereas doing it per minor would
/// put the whole table back on the minor path.
pub fn prune_ephemeron_tables(classify: &mut dyn FnMut(usize) -> Option<usize>) {
    let pruners = {
        let guard = EPHEMERON_PRUNERS.read();
        *guard
    };
    for slot in pruners.iter().flatten() {
        slot(classify);
    }
}

/// Signature expected by [`register_young_owner_reconciler`].
///
/// Same classifier contract as [`EphemeronPrunerFn`], asked at the end of a
/// minor collection and only about the entries whose owner was young when the
/// entry was made.
pub type YoungOwnerReconcilerFn = fn(&mut dyn FnMut(usize) -> Option<usize>);

static YOUNG_OWNER_RECONCILERS: parking_lot::RwLock<
    [Option<YoungOwnerReconcilerFn>; MAX_EPHEMERON_PRUNERS],
> = parking_lot::RwLock::new([None; MAX_EPHEMERON_PRUNERS]);

/// Register a side table that keys entries by the address of an owner that may
/// still be in the nursery.
///
/// A nursery address is reused as soon as the collection that dropped it
/// resets the nursery, so an entry whose owner dies young does not merely leak
/// — the next object allocated at that address answers to it.  A major-only
/// pruner cannot close that window, hence this second registration.
///
/// Duplicate registrations are tolerated — the reconciler is only appended if
/// not already present.
pub fn register_young_owner_reconciler(reconciler: YoungOwnerReconcilerFn) {
    let mut guard = YOUNG_OWNER_RECONCILERS.write();
    for slot in guard.iter_mut() {
        match slot {
            Some(existing) if std::ptr::fn_addr_eq(*existing, reconciler) => return,
            None => {
                *slot = Some(reconciler);
                return;
            }
            _ => {}
        }
    }
    panic!(
        "register_young_owner_reconciler: capacity exceeded ({MAX_EPHEMERON_PRUNERS} reconcilers \
         already registered)"
    );
}

/// Invoke every registered reconciler with a classifier that resolves a young
/// owner address to where it moved, or reports that it died.
///
/// Called from a minor collection once every live nursery object has been
/// forwarded out and while the forwarding headers are still readable — the
/// same point `invalidate_young_weakrefs` settles the weakref slots.  Unlike
/// [`prune_ephemeron_tables`] this costs O(entries made since the previous
/// minor), not O(table).
pub fn reconcile_young_owner_tables(classify: &mut dyn FnMut(usize) -> Option<usize>) {
    let reconcilers = {
        let guard = YOUNG_OWNER_RECONCILERS.read();
        *guard
    };
    for slot in reconcilers.iter().flatten() {
        slot(classify);
    }
}

// ── Extern "C" interface for compiled code ──────────────────────

/// Push a GC reference from compiled code.
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_push(gcref_raw: i64) -> i64 {
    let gcref = GcRef(gcref_raw as usize);
    push(gcref) as i64
}

/// Pop shadow stack to depth and return the value at given index.
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_pop_and_get(_depth: i64, index: i64) -> i64 {
    SHADOW_STACK.with(|ss| {
        let idx = index as usize;
        if idx < ss.len() {
            ss.get(idx).0 as i64
        } else {
            0
        }
    })
}

/// Set shadow stack depth (truncate).
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_set_depth(new_depth: i64) {
    SHADOW_STACK.with(|ss| ss.truncate(new_depth as usize));
}

/// Read the top jf_ptr from the jitframe shadow stack.
///
/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]; MOV ebp, [ecx - WORD]
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_get_top_jf_ptr() -> i64 {
    jf_top_ptr().0 as i64
}

/// _call_header_shadowstack (assembler.py:1122-1128).
/// Non-compiled callers use this; compiled code uses inline MOVs.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_push(jf_ptr_raw: i64) -> i64 {
    push_jf(crate::GcRef(jf_ptr_raw as usize)) as i64
}

/// _call_footer_shadowstack (assembler.py:1130-1136).
/// Non-compiled callers use this; compiled code uses inline SUB.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_pop_to(depth: i64) {
    pop_jf_to(depth as usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    // Keep these tests serialized because some of them intentionally shrink
    // the current thread's capacity and then re-raise an expected panic.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn cached_root_stack_slot_reads_the_same_top() {
        let _lock = TEST_MUTEX.lock();
        let cached = Cell::new(0usize);
        let depth = depth();
        push(GcRef(0x1111));
        cache_root_stack_slot(&cached);
        let raw = cached.get();
        assert_ne!(raw, 0);
        cache_root_stack_slot(&cached);
        assert_eq!(cached.get(), raw);
        let slot = slot_from_cached(raw);
        assert_eq!(unsafe { slot_top_ref(slot) }, top_ref());
        assert_eq!(unsafe { slot_top_ref(slot) }.0, 0x1111);
        pop_to(depth);
    }

    #[test]
    fn resume_roots_preserve_pointer_fields_and_register_stride() {
        let _lock = TEST_MUTEX.lock();
        #[repr(C, align(8))]
        struct Fields([usize; 3]);
        let mut fields = Fields([0x1111, 0x2000, 0x3333]);
        let mut registers = [0x4000_i64, 0x5000];
        let depth = resume_ref_roots_depth();
        // On wasm32 this field is only four-byte aligned and has a live
        // neighbouring word: neither an i64 reference nor slice is valid.
        unsafe {
            push_resume_ref_root(&mut *(&mut fields.0[1] as *mut usize).cast::<GcRef>());
            push_resume_ref_roots(&mut registers);
        }
        assert!(resume_ref_slice_registered(registers.as_ptr()));
        walk_resume_ref_roots(|slot| slot.0 += 0x10);
        pop_resume_ref_roots_to(depth);
        assert_eq!(fields.0, [0x1111, 0x2010, 0x3333]);
        assert_eq!(registers, [0x4010, 0x5010]);
    }

    fn jf_root_stack_base_for_test() -> usize {
        JF_ROOT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.ensure_init();
            stack.base
        })
    }

    fn jf_root_stack_capacity_for_test() -> usize {
        JF_ROOT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.ensure_init();
            stack.capacity
        })
    }

    fn set_jf_root_stack_capacity_for_test(capacity: usize) {
        JF_ROOT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.ensure_init();
            stack.capacity = capacity;
        });
    }

    #[test]
    fn test_push_pop_roundtrip() {
        let _lock = TEST_MUTEX.lock();
        clear();
        let a = GcRef(0x1000);
        let b = GcRef(0x2000);
        let depth = push(a);
        assert_eq!(depth, 0);
        push(b);
        assert_eq!(super::depth(), 2);

        assert_eq!(super::get(0), a);
        assert_eq!(super::get(1), b);

        pop_to(depth);
        assert_eq!(super::depth(), 0);
    }

    #[test]
    #[should_panic(expected = "shadow stack pop_to")]
    fn test_pop_to_above_depth_panics() {
        let _lock = TEST_MUTEX.lock();
        clear();
        push(GcRef(0x1000));
        // An unbalanced scope must not silently no-op: truncating past the
        // top would strand the roots below it.
        pop_to(5);
    }

    #[test]
    fn test_walk_roots_updates() {
        let _lock = TEST_MUTEX.lock();
        clear();
        push(GcRef(0x1000));
        push(GcRef(0x2000));

        // Simulate GC moving objects
        walk_roots(|gcref| {
            gcref.0 += 0x100;
        });

        assert_eq!(get(0), GcRef(0x1100));
        assert_eq!(get(1), GcRef(0x2100));
        clear();
    }

    #[test]
    fn test_shadow_stack_slot_forwards_and_nests() {
        let _lock = TEST_MUTEX.lock();
        clear();
        let outer = shadow_stack_slot();
        let outer_depth = outer.depth();
        let first = outer.push(GcRef(0x1000));
        assert_eq!(first, outer_depth);
        {
            let inner = shadow_stack_slot();
            let inner_depth = inner.depth();
            inner.push(GcRef(0x2000));
            // A moving collection writes the forwarded address back into the slot.
            walk_roots(|gcref| gcref.0 += 0x10);
            assert_eq!(inner.get(inner_depth), GcRef(0x2010));
            inner.pop_to(inner_depth);
        }
        assert_eq!(outer.get(first), GcRef(0x1010));
        assert_eq!(outer.depth(), outer_depth + 1);
        outer.pop_to(outer_depth);
        assert_eq!(outer.depth(), outer_depth);
    }

    #[test]
    fn test_owner_roots_have_independent_fixed_slots() {
        let _lock = TEST_MUTEX.lock();
        clear();
        let lexical = push(GcRef(0x1000));
        let first = acquire_owner_root(GcRef(0x2000));
        let second = acquire_owner_root(GcRef(0x3000));

        pop_to(lexical);
        assert_eq!(super::depth(), 0);
        let mut walked = Vec::new();
        walk_roots(|root| walked.push(*root));
        assert_eq!(walked, vec![GcRef(0x2000), GcRef(0x3000)]);

        release_owner_root(first);
        release_owner_root(second);
        walked.clear();
        walk_roots(|root| walked.push(*root));
        assert!(walked.is_empty());
    }

    #[test]
    fn test_mutator_extra_area_walks_current_thread() {
        unsafe fn walk_cell(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
            let slot = unsafe { &mut *(data as *mut GcRef) };
            visitor(slot);
        }

        let _lock = TEST_MUTEX.lock();
        let mut root = GcRef(0x1000);
        register_mutator();
        // SAFETY: `root` remains valid until unregistration, and `walk_cell`
        // derives its sole dereference from the supplied `data` pointer.
        unsafe {
            register_mutator_extra_area(
                walk_cell,
                &mut root as *mut GcRef as *const (),
                "test_cell",
            );
        }
        walk_my_extra_areas(|gcref| gcref.0 += 0x100);
        unregister_mutator();
        assert_eq!(root, GcRef(0x1100));
    }

    #[test]
    fn scoped_extra_areas_forward_and_retire_on_drop_or_unwind() {
        unsafe fn walk_cell(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
            let slot = unsafe { &*(data as *const Cell<GcRef>) };
            let mut value = slot.get();
            visitor(&mut value);
            slot.set(value);
        }
        let _lock = TEST_MUTEX.lock();
        let outer = Cell::new(GcRef(0x1000));
        let inner = Cell::new(GcRef(0x2000));
        register_mutator();
        let outer_guard = unsafe {
            MutatorExtraAreaGuard::new(walk_cell, &outer as *const _ as *const (), "outer")
        };
        let inner_guard = unsafe {
            MutatorExtraAreaGuard::new(walk_cell, &inner as *const _ as *const (), "inner")
        };
        let mut seen = Vec::new();
        walk_my_extra_areas(|root| {
            seen.push(root.0);
            root.0 += 0x80;
        });
        assert_eq!(seen, [0x1000, 0x2000]);
        assert_eq!(outer.get(), GcRef(0x1080));
        assert_eq!(inner.get(), GcRef(0x2080));
        drop(outer_guard);
        seen.clear();
        walk_my_extra_areas(|root| seen.push(root.0));
        assert_eq!(seen, [0x2080]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = inner_guard;
            panic!("test unwind");
        }));
        assert!(result.is_err());
        walk_my_extra_areas(|_| panic!("retired root still visible"));
        unregister_mutator();
    }

    #[test]
    fn permanent_extra_area_survives_unregister() {
        unsafe fn walk_cell(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
            let slot = unsafe { &*(data as *const Cell<GcRef>) };
            let mut value = slot.get();
            visitor(&mut value);
            slot.set(value);
        }
        let _lock = TEST_MUTEX.lock();
        let cell = Cell::new(GcRef(0x1000));
        register_mutator();
        let guard = unsafe {
            MutatorExtraAreaGuard::permanent(
                walk_cell,
                &cell as *const _ as *const (),
                "permanent_cell",
            )
        };
        walk_my_extra_areas(|root| root.0 += 1);
        assert_eq!(cell.get(), GcRef(0x1001));
        unregister_mutator();
        drop(guard);
    }

    /// A thread that unregisters leaves nothing behind for the all-mutator
    /// walk to reach, and a later thread can take its place.
    ///
    /// `thread_die` (`shadowstack.py`) drops the dying thread's entry
    /// from `gcdata.thread_stacks` before the last GIL release, so no
    /// subsequent collection walks a stack whose owner is gone. The registry
    /// here is that table; `register_mutator` asserts a thread is not present
    /// twice, so a second registration succeeding is what proves the first
    /// entry was actually removed rather than merely emptied.
    #[test]
    fn test_mutator_teardown_releases_its_registry_entry() {
        let _lock = TEST_MUTEX.lock();
        std::thread::spawn(|| {
            register_mutator();
            {
                let root = OwnerRootGuard::new(GcRef(0x4000));
                assert_eq!(root.get(), GcRef(0x4000));
            }
            unregister_mutator();
            // The entry is gone, so this thread can register again.
            register_mutator();
            unregister_mutator();
        })
        .join()
        .expect("the registered thread must tear down without panicking");
    }

    #[test]
    fn test_extern_c_interface() {
        let _lock = TEST_MUTEX.lock();
        clear();
        let depth = majit_shadow_stack_push(0x3000);
        assert_eq!(depth, 0);
        majit_shadow_stack_push(0x4000);

        let val = majit_shadow_stack_pop_and_get(0, 0);
        assert_eq!(val, 0x3000);
        let val = majit_shadow_stack_pop_and_get(0, 1);
        assert_eq!(val, 0x4000);

        majit_shadow_stack_set_depth(0);
        assert_eq!(super::depth(), 0);
    }

    #[test]
    fn test_jf_shadow_stack_push_pop() {
        let _lock = TEST_MUTEX.lock();
        clear();
        assert_eq!(jf_depth(), 0);
        let depth = push_jf(GcRef(0x1000));
        assert_eq!(depth, 0);
        assert_eq!(jf_depth(), 1);
        assert_eq!(peek_jf(0), GcRef(0x1000));
        push_jf(GcRef(0x2000));
        assert_eq!(jf_depth(), 2);
        assert_eq!(peek_jf(1), GcRef(0x2000));
        pop_jf_to(depth);
        assert_eq!(jf_depth(), 0);
    }

    #[test]
    fn test_walk_jf_roots_updates_gcref() {
        let _lock = TEST_MUTEX.lock();
        clear();
        push_jf(GcRef(0x1000));
        push_jf(GcRef(0x2000));

        // Simulate GC moving jitframes (root_walker semantics)
        walk_jf_roots(|gcref| {
            gcref.0 += 0x100;
        });

        // After walk, entries should be updated in-place
        assert_eq!(peek_jf(0), GcRef(0x1100));
        assert_eq!(jf_top_ptr(), GcRef(0x2100));
        pop_jf_to(0);
    }

    #[test]
    fn test_jf_top_ptr_reload() {
        let _lock = TEST_MUTEX.lock();
        clear();
        push_jf(GcRef(0xABCD));
        assert_eq!(jf_top_ptr(), GcRef(0xABCD));
        assert_eq!(majit_jf_shadow_stack_get_top_jf_ptr(), 0xABCD);

        // Simulate GC updating the entry
        walk_jf_roots(|gcref| {
            gcref.0 = 0xDEAD;
        });
        assert_eq!(jf_top_ptr(), GcRef(0xDEAD));
        assert_eq!(majit_jf_shadow_stack_get_top_jf_ptr(), 0xDEAD_i64);
        pop_jf_to(0);
    }

    #[test]
    fn test_jf_flat_array_layout() {
        let _lock = TEST_MUTEX.lock();
        // Verify the flat array layout matches RPython's [is_minor, jf_ptr] pairs
        clear();
        push_jf(GcRef(0xAAAA));
        push_jf(GcRef(0xBBBB));
        unsafe {
            let base = jf_root_stack_base_for_test() as *const usize;
            // Entry 0: [1, 0xAAAA]
            assert_eq!(*base, 1);
            assert_eq!(*base.add(1), 0xAAAA);
            // Entry 1: [1, 0xBBBB]
            assert_eq!(*base.add(2), 1);
            assert_eq!(*base.add(3), 0xBBBB);
        }
        pop_jf_to(0);
    }

    /// rpython/memory/gctransform/shadowstack.py increase_root_stack_depth parity:
    /// sys.setrecursionlimit(>1000) must resize the jitframe
    /// shadow-stack backing buffer, not just raise a limit.
    #[test]
    fn test_increase_root_stack_depth_grows_backing_buffer() {
        let _lock = TEST_MUTEX.lock();
        clear();
        // Populate a known jf_ptr that must survive the resize.
        push_jf(GcRef(0xDEADBEEF));
        let base_before = jf_root_stack_base_for_test();
        let capacity_before = jf_root_stack_capacity_for_test();
        let limit_addr = get_root_stack_limit_addr();
        let limit_before = unsafe { *(limit_addr as *const usize) };
        assert_eq!(capacity_before, DEFAULT_SHADOW_STACK_DEPTH);

        // Grow to 2x the default. RPython's resize copies the used
        // portion to the new buffer.
        let new_cap = DEFAULT_SHADOW_STACK_DEPTH * 2;
        increase_root_stack_depth(new_cap);

        let base_after = jf_root_stack_base_for_test();
        let capacity_after = jf_root_stack_capacity_for_test();
        assert_eq!(get_root_stack_limit_addr(), limit_addr);
        assert_ne!(unsafe { *(limit_addr as *const usize) }, limit_before);
        assert_eq!(
            unsafe { *(limit_addr as *const usize) },
            base_after + new_cap * 2 * WORD,
            "the stable limit cell must track the reallocated backing buffer"
        );
        assert_ne!(
            base_before, base_after,
            "resize must reallocate the backing buffer"
        );
        assert_eq!(
            capacity_after, new_cap,
            "capacity atomic must reflect the new size"
        );
        // The copied jf_ptr must still be reachable via peek_jf.
        assert_eq!(peek_jf(0), GcRef(0xDEADBEEF));
        // Pushing up to the new capacity must not overflow.
        for i in 1..new_cap {
            push_jf(GcRef(0x1000 + i));
        }
        assert_eq!(jf_depth(), new_cap);
        pop_jf_to(0);
    }

    /// Pushing exactly `capacity` entries is OK; push #{capacity+1}
    /// must trip the overflow assert, not silently corrupt heap.
    #[test]
    #[should_panic(expected = "jf shadow stack overflow")]
    fn test_jf_shadow_stack_overflow_panics() {
        let lock = TEST_MUTEX.lock();
        clear();
        // Pre-shrink: the capacity atomic drives the assert; the
        // backing HEAP buffer is larger than 4 entries, so the
        // overflow fires before any OOB write.
        let original_cap = jf_root_stack_capacity_for_test();
        set_jf_root_stack_capacity_for_test(4);
        let result = std::panic::catch_unwind(|| {
            for i in 0..4 {
                push_jf(GcRef(i));
            }
            push_jf(GcRef(99)); // MUST panic
        });
        // Restore state before dropping the lock so the next test
        // sees the default capacity. Then release the lock cleanly
        // before re-raising, so the Mutex doesn't poison.
        pop_jf_to(0);
        set_jf_root_stack_capacity_for_test(original_cap);
        drop(lock);
        match result {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(_) => panic!("jf shadow stack overflow did not fire"),
        }
    }
}
