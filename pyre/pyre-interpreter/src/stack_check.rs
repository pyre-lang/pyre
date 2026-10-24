//! Stack overflow protection — RPython rstack parity.
//!
//! Port of `rpython/rlib/rstack.py stack_check()` +
//! `rpython/translator/c/src/stack.c:25 LL_stack_too_big_slowpath`. The
//! four primary entrypoints match RPython's `LL_stack_*` C ABI so the
//! JIT backends can call them from emitted code:
//!
//!   * [`pyre_stack_get_end`] — captured stack base (high address; stacks
//!     grow downward). Raw read, returns 0 when never captured.
//!   * [`pyre_stack_get_length`] — effective stack budget in bytes.
//!   * [`pyre_stack_set_length_fraction`] — multiply `MAX_STACK_SIZE` by
//!     `frac`, clamp to the OS stack, and store the result as the new
//!     effective length.
//!   * [`pyre_stack_too_big_slowpath`] — four-case slow path: first-time
//!     capture, thread-switch cache refresh, stack-underflow base
//!     revision, real overflow.
//!
//! The JIT-addressable state lives in the global [`PYRE_STACKTOOBIG`]
//! matching RPython's `rpy_stacktoobig_t` struct (`stack.h:23`), so
//! `pyre_stack_get_end_adr` / `pyre_stack_get_length_adr` can hand out
//! stable addresses for backend-emitted inline probes. The source of
//! truth for `stack_end` however is a thread-local mirror
//! [`TL_STACK_END`] matching `pypy_threadlocal_s::stack_end`
//! (`stack.c:30`); the slow path reads the TLS value first and refreshes
//! the global cache from it on a thread switch (`stack.c:49`), then
//! writes both when the base is newly captured or revised
//! (`stack.c:63-64`). The requested stack length remains process-global;
//! only its safety clamp uses the current thread's actual OS stack size.
//! That is genuine thread state, and the slow path refreshes the same
//! backend-visible global cache that RPython uses.
//!
//! `sys.setrecursionlimit(N)` calls `pyre_stack_set_length_fraction(N *
//! 0.001)` and stores `N` as the visible recursion limit. The default
//! recursion limit is 1000, matching CPython/PyPy.
//!
//! [`stack_check`] is the Rust-side inline fast path. The JIT prologue
//! emits the equivalent inline sequence directly (see
//! `rpython/jit/backend/x86/assembler.py _call_header_with_stack_check
//! _call_header_with_stack_check`).

use std::sync::atomic::{AtomicI32, AtomicU8, AtomicUsize, Ordering};

use pyre_object::interp_exceptions::{ExcKind, w_exception_new};

use crate::PyError;

/// Default recursion limit, re-exported from `crate::module::sys::state`
/// where the visible limit actually lives. The storage is owned by the
/// sys module (matching `space.sys.recursionlimit` in
/// `pypy/module/sys/moduledef.py:25`); `stack_check` only reads and
/// writes it via the accessor functions.
pub use crate::module::sys::state::{DEFAULT_RECURSION_LIMIT, MAX_RECURSION_LIMIT};

/// rpython/translator/c/src/stack.h:8-19 `MAX_STACK_SIZE` adaptation.
///
/// stack.h picks the constant per architecture for exactly one reason:
/// 768 KB "is only enough for 406 levels on ppc64", so platforms whose
/// frames are bigger get the larger budget instead. pyre applies the
/// same rule to its own frames. The budget is scaled by `recursionlimit
/// / 1000` ([`pyre_stack_set_length_fraction`]), so this constant is
/// exactly "bytes for 1000 Python call levels" and has to cover the
/// most expensive level pyre can produce.
///
/// A level the interpreter runs costs ~1.7 KB of native stack
/// (`funccall_valuestack` → `OpcodeStepExecutor::call` → `eval_loop_jit`
/// → `eval_with_jit`). A level that enters compiled code and leaves it
/// through a guard failure costs ~6.3 KB, because the resume chain
/// nests on the native stack once per level:
/// `call_assembler_helper_trampoline` → `jit_blackhole_resume_from_guard`
/// → `blackhole_resume_via_rd_numb` → `BlackholeInterpreter::run` →
/// `handler_residual_call_r_r` → `bh_call_fn`. `48 << 18` is twice what
/// 1000 such levels need, which keeps the byte budget above the
/// recursion limit's own cutoff: `sys.setrecursionlimit(N)` is what
/// bounds Python call depth and the byte budget stays the hard-stack
/// guard it is upstream.
///
/// Sized against the interpreter alone (`11 << 18`), the guard fired at
/// 458 levels as soon as a recursive function went hot — under half the
/// default limit of 1000, and reached by nothing more than calling the
/// same recursive function five times.
///
/// `wasm32` keeps `3 << 18`: its linear-memory stack is sized at link
/// time (1 MB by default) with no guard page and no `getrlimit` for
/// [`pyre_stack_set_length_fraction`] to clamp against, so a budget
/// wider than the real stack would corrupt memory instead of raising.
#[cfg(target_arch = "wasm32")]
pub const MAX_STACK_SIZE: usize = 3 << 18;
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_STACK_SIZE: usize = 48 << 18;

/// Native stack every thread that runs Python code is budgeted against.
///
/// Rust's platform default can be as small as 2 MiB, which cannot hold pyre's
/// translated Rust frame shape even for stdlib recursion such as
/// `functools.lru_cache`'s `fib(100)`.  `_thread.stack_size()` still
/// reports/configures the Python-visible value; zero selects this
/// implementation default.
///
/// [`effective_stack_length`] leaves a quarter of the stack to the Rust/C
/// frames between the probe and the guard page.  Most hosts use 20 MiB (a
/// 15 MiB ceiling). Windows uses 24 MiB (an 18 MiB ceiling): Cranelift's Win64
/// prologue/resume frames need about 16.8 MiB to reach Python's logical cutoff
/// at a recursion limit of 2000.  Keeping the native guard above that cutoff
/// makes the Python 3.14 logical-depth check decide the observable limit, while
/// the native guard remains the hard-stop PyPy's `rstack` design requires.
/// Both stay far below the 256 MiB interpreter stack reserved by
/// `pyrex::main_entry`.  These ceilings cover the measured guard-resume cost
/// without making
/// `support.infinite_recursion(20_000)` consume the much larger interpreter
/// stack before its native guard fires.  At 8 MiB the clamp cut a thread's
/// budget below what the *default* limit is worth, so the guard, not the limit,
/// decided how deep a thread could recurse; at 64 MiB highly nested pure-Python
/// decoders spent minutes below the clamp and every default worker reserved an
/// excessive amount of address space.
///
/// The thread the interpreter starts on announces this too, though the stack
/// it runs on is much larger (`pyrex::main_entry`).  `stack_length` is
/// process-global — the JIT's inline probe reads its address as an immediate
/// (`pyre_stack_get_length_adr`) — so whatever a thread stores there is what
/// every other thread's fast path reads until one of them takes the slow
/// path.  A single figure for all Python-running threads keeps that shared
/// word meaningful; upstream gets the same uniformity for free because
/// `_ll_stack_os_limit` reads `getrlimit`, which is process-wide.
#[cfg(windows)]
pub const DEFAULT_RUNTIME_THREAD_STACK_SIZE: usize = 24 * 1024 * 1024;
#[cfg(not(windows))]
pub const DEFAULT_RUNTIME_THREAD_STACK_SIZE: usize = 20 * 1024 * 1024;

/// Process-wide requested byte budget, corresponding to RPython's
/// `MAX_STACK_SIZE * recursionlimit / 1000`.  This is semantic configuration
/// and therefore deliberately global.  Only the OS stack capacity used to
/// clamp it is thread-local.
static REQUESTED_STACK_LENGTH: AtomicUsize = AtomicUsize::new(MAX_STACK_SIZE);

/// rpython/translator/c/src/stack.h:23-27 `rpy_stacktoobig_t` parity.
///
/// ```c
/// typedef struct {
///     char *stack_end;
///     Signed stack_length;
///     char report_error;
/// } rpy_stacktoobig_t;
/// ```
///
/// The field order and `#[repr(C)]` layout exactly match the C struct,
/// so an external tool that knows `rpy_stacktoobig_t` can reinterpret
/// `PYRE_STACKTOOBIG` byte-for-byte. Individual fields use
/// `AtomicUsize` / `AtomicU8` so safe Rust can load and store without
/// `unsafe`; the generated machine code (with `Ordering::Relaxed`) is
/// identical to a plain memory access on aligned naturally-sized
/// fields, matching the plain C loads/stores `rpy_stacktoobig` sees.
#[repr(C)]
pub struct PyreStackTooBig {
    pub stack_end: AtomicUsize,
    pub stack_length: AtomicUsize,
    pub report_error: AtomicU8,
}

/// rpython/translator/c/src/stack.c:14-18 `rpy_stacktoobig` parity.
/// Globally-addressable backing store for the fast-path stack check.
/// JIT backends take `&raw const PYRE_STACKTOOBIG.stack_end` /
/// `&raw const PYRE_STACKTOOBIG.stack_length` via
/// [`pyre_stack_get_end_adr`] / [`pyre_stack_get_length_adr`] and emit
/// inline `MOV [endaddr]; SUB rsp; CMP [lengthaddr]` probes in the
/// compiled prologue (see `rpython/jit/backend/x86/assembler.py:1085`).
///
/// This is the global cache only; the authoritative `stack_end` lives
/// in the thread-local [`TL_STACK_END`] (stack.c:30-40). The inline
/// probe reads the global to avoid the cost of a TLS load per call,
/// and the slow path reconciles the two (stack.c:49, 63-64).
#[unsafe(no_mangle)]
pub static PYRE_STACKTOOBIG: PyreStackTooBig = PyreStackTooBig {
    stack_end: AtomicUsize::new(0),
    stack_length: AtomicUsize::new(MAX_STACK_SIZE),
    report_error: AtomicU8::new(1),
};

thread_local! {
    /// `stack.c:30` `struct pypy_threadlocal_s *tl1; baseptr = tl1->stack_end;`
    /// parity — per-thread mirror of the captured stack base, source
    /// of truth for `LL_stack_too_big_slowpath`. The global
    /// `PYRE_STACKTOOBIG.stack_end` is a cache refreshed from this
    /// value on a thread switch (stack.c:49) and written alongside
    /// the TLS on first-time capture / underflow revision
    /// (stack.c:63-64).
    static TL_STACK_END: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Capacity of the current OS thread's native stack.  Unlike type/module
    /// state, this is intrinsically thread-specific.  A zero value means
    /// "query the platform limit lazily"; Python-created threads install the
    /// exact size supplied to `std::thread::Builder`.
    static TL_OS_STACK_SIZE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// stack.c:18 `report_error`, mirrored per thread for free-threaded
    /// critical-code suppression. The global struct field remains the
    /// stable backend-visible cache; the slow path reads this source of truth.
    static TL_REPORT_ERROR: std::cell::Cell<u8> = const { std::cell::Cell::new(1) };

    /// Exception parked by [`park_jit_pending_error`] for the current OS
    /// thread. `unpackiterable_driver` is the producer: its merge-point hook
    /// returns `()` and the error is re-raised at the next
    /// [`drain_jit_pending_exception`]. The prologue stack check does not
    /// write this cell; `stack_check_slowpath` publishes through
    /// `jit_publish_exception` (`llmodel.py` `_store_exception`).
    static TL_JIT_PENDING_EXCEPTION: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Serialize tests that mutate the process-global stack-probe state with
    /// tests that execute compiled traces reading that state.
    ///
    /// The Rust test harness runs both groups in parallel. Keeping separate
    /// locks lets a stack-check test plant a synthetic `stack_end` or
    /// `stack_length` while a Cranelift prologue is probing the real thread
    /// stack, producing a spurious overflow. This lock also remains the
    /// single-threaded ownership boundary for the process-global JIT compiler
    /// state.
    static STACK_AND_JIT_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// Acquire the shared test lock.  `parking_lot` has no poison flag, so one
    /// failed test does not cascade into the rest of the test binary.
    pub(crate) fn stack_and_jit_test_guard() -> parking_lot::MutexGuard<'static, ()> {
        STACK_AND_JIT_TEST_LOCK.lock()
    }
}

// The visible recursion limit lives in `crate::module::sys::state`
// (`space.sys.recursionlimit` parity, `pypy/module/sys/vm.py:96`);
// `stack_check` forwards reads/writes through
// `crate::module::sys::state::{recursion_limit, set_recursion_limit}`.

/// Store the exception [`park_jit_pending_error`] parks for the current
/// thread. A value of 0 means the slot is empty.
#[inline]
fn set_jit_pending_exception(obj: pyre_object::PyObjectRef) {
    TL_JIT_PENDING_EXCEPTION.with(|slot| slot.set(obj as i64));
}

/// Test/runtime helper that clears only the current thread's pending
/// JIT exception slot.
#[inline]
#[allow(dead_code)]
fn clear_jit_pending_exception() {
    TL_JIT_PENDING_EXCEPTION.with(|slot| slot.set(0));
}

/// Read an approximation of the current stack pointer by taking the
/// address of a stack-allocated local. The exact value depends on the
/// caller's frame layout but is monotonically decreasing as recursion
/// deepens, which is all `stack_check` needs. Equivalent to RPython's
/// `llop.stack_current()`.
#[inline(always)]
fn current_sp() -> usize {
    let probe: usize = 0;
    &probe as *const usize as usize
}

/// Reset the captured stack base to the current SP.
///
/// Call this once at the outermost interpreter entry (e.g. just
/// before running a Python program) so subsequent `stack_check`
/// calls measure depth relative to that point. If the base is never
/// reset explicitly, the first `stack_check` call lets
/// [`pyre_stack_too_big_slowpath`] capture it lazily (first-time path
/// in `stack.c:42-44`).
pub fn reset_stack_base() {
    // stack.c:63-64 parity: update both the per-thread mirror and
    // the global cache. TL_STACK_END is the source of truth; the
    // global is refreshed so the inline fast-path probe sees the
    // new value without a TLS load.
    let sp = current_sp();
    TL_STACK_END.with(|c| c.set(sp));
    PYRE_STACKTOOBIG.stack_end.store(sp, Ordering::Relaxed);
    PYRE_STACKTOOBIG
        .stack_length
        .store(effective_stack_length(), Ordering::Relaxed);
}

/// Install the actual native stack size for the current runtime thread.
///
/// This is called at the outermost entry of Python-created threads.  Passing
/// zero leaves platform discovery in charge, matching the main-thread path.
pub fn configure_current_thread_stack_size(size: usize) {
    TL_OS_STACK_SIZE.with(|slot| slot.set(size));
    reset_stack_base();
}

/// `stack.c _ll_stack_os_limit`. Size in bytes of this thread's C stack as
/// the OS sees it, or 0 when unknown/unlimited. Computed once and cached:
/// this is not on a hot path, but `test.support.infinite_recursion()`
/// toggles the recursion limit in a loop, so caching keeps repeated calls
/// free.
#[cfg(not(target_arch = "wasm32"))]
fn stack_os_limit() -> usize {
    static CACHED: AtomicUsize = AtomicUsize::new(usize::MAX);
    let cached = CACHED.load(Ordering::Relaxed);
    if cached != usize::MAX {
        return cached;
    }
    let limit = query_stack_os_limit();
    CACHED.store(limit, Ordering::Relaxed);
    limit
}

/// The `_WIN32` arm: `GetCurrentThreadStackLimits` (Windows 8 and later,
/// which every target this builds for is) reports the reserved range of the
/// calling thread's stack.
#[cfg(windows)]
fn query_stack_os_limit() -> usize {
    let mut low = 0usize;
    let mut high = 0usize;
    unsafe {
        windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut low, &mut high)
    };
    high - low
}

/// The `getrlimit(RLIMIT_STACK)` arm.
#[cfg(not(any(windows, target_arch = "wasm32")))]
fn query_stack_os_limit() -> usize {
    let mut rl: libc::rlimit = unsafe { std::mem::zeroed() };
    match unsafe { libc::getrlimit(libc::RLIMIT_STACK, &mut rl) } {
        0 if rl.rlim_cur != libc::RLIM_INFINITY && rl.rlim_cur != 0 => rl.rlim_cur as usize,
        _ => 0,
    }
}

/// wasm32 has no runtime query, so no clamp applies — the `#ifdef`
/// fallthrough of `_ll_stack_os_limit`.
#[cfg(target_arch = "wasm32")]
fn stack_os_limit() -> usize {
    0
}

/// Raise the soft `RLIMIT_STACK` towards `size`, returning the stack the
/// thread that started the process can now grow to.
///
/// That thread's stack is not sized once at exec: the kernel grows it on
/// demand up to the current soft limit, so raising the limit is what makes
/// room for the interpreter.  The request is clamped to the hard limit rather
/// than abandoned when it does not fit — darwin's stops at 65520 KiB, a
/// hair under 64 MiB, and refusing there would leave the interpreter on the
/// default 8 MiB.  `usize::MAX` reports an unlimited stack, which is what
/// [`stack_os_limit`] declines to describe.
#[cfg(unix)]
pub fn raise_main_thread_stack_limit(size: usize) -> usize {
    let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    if unsafe { libc::getrlimit(libc::RLIMIT_STACK, limit.as_mut_ptr()) } != 0 {
        return 0;
    }
    let mut limit = unsafe { limit.assume_init() };
    if limit.rlim_cur == libc::RLIM_INFINITY {
        return usize::MAX;
    }
    let mut wanted = size as libc::rlim_t;
    if limit.rlim_max != libc::RLIM_INFINITY {
        wanted = wanted.min(limit.rlim_max);
    }
    let granted = if wanted > limit.rlim_cur {
        let previous = limit.rlim_cur;
        limit.rlim_cur = wanted;
        // A refused `setrlimit` changes nothing, so the limit read above is
        // still what the process has.  Reporting the request instead would
        // buy a byte budget the stack cannot hold, and the guard would sit
        // past the guard page.
        if unsafe { libc::setrlimit(libc::RLIMIT_STACK, &limit) } == 0 {
            wanted
        } else {
            previous
        }
    } else {
        limit.rlim_cur
    };
    usize::try_from(granted).unwrap_or(usize::MAX)
}

#[cfg(not(unix))]
pub fn raise_main_thread_stack_limit(_size: usize) -> usize {
    0
}

/// RPython's stack.c safety clamp: retain one quarter of the native stack for
/// Rust/C frames between the probe and the guard page.
fn effective_stack_length() -> usize {
    let requested = REQUESTED_STACK_LENGTH.load(Ordering::Relaxed);
    let os_size = TL_OS_STACK_SIZE.with(|slot| {
        let installed = slot.get();
        if installed == 0 {
            stack_os_limit()
        } else {
            installed
        }
    });
    if os_size == 0 {
        requested
    } else {
        requested.min(os_size.saturating_sub(os_size >> 2))
    }
}

/// Test-only helper that plants a synthetic `stack_end` into both the
/// TLS mirror (stack.c:40 source of truth) and the global cache
/// (stack.c:14,49 inline-probe visible). The slowpath reads the TLS,
/// so tests that want to force an overflow decision must write both —
/// writing the global alone is silently overridden by the TLS load
/// inside `pyre_stack_too_big_slowpath`.
pub fn plant_stack_end_for_tests(value: usize) {
    TL_STACK_END.with(|c| c.set(value));
    PYRE_STACKTOOBIG.stack_end.store(value, Ordering::Relaxed);
}

/// rpython/translator/c/src/stack.h:37 `LL_stack_get_end` parity.
/// Raw read of `rpy_stacktoobig.stack_end`. Returns `0` when the base
/// has never been captured — the fast-path compare will miss on the
/// `end.wrapping_sub(current) > length` condition and dispatch to
/// [`pyre_stack_too_big_slowpath`], which performs first-time capture.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_end() -> usize {
    PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed)
}

/// rpython/translator/c/src/stack.h:38 `LL_stack_get_length` parity.
/// Raw read of `rpy_stacktoobig.stack_length`.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_length() -> usize {
    PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed)
}

/// rpython/translator/c/src/stack.h:39 `LL_stack_get_end_adr` parity.
/// Returns the stable address of `PYRE_STACKTOOBIG.stack_end` for the
/// JIT backend to emit inline `MOV reg, [endaddr]` loads. Mirrors
/// `rpython/rlib/rstack.py _stack_get_end_adr`.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_end_adr() -> usize {
    &raw const PYRE_STACKTOOBIG.stack_end as usize
}

/// rpython/translator/c/src/stack.h:40 `LL_stack_get_length_adr` parity.
/// Returns the stable address of `PYRE_STACKTOOBIG.stack_length` for
/// the JIT backend to emit inline `CMP reg, [lengthaddr]` compares.
/// Mirrors `rpython/rlib/rstack.py _stack_get_length_adr`.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_get_length_adr() -> usize {
    &raw const PYRE_STACKTOOBIG.stack_length as usize
}

/// rpython/translator/c/src/stack.c:58-77 `LL_stack_set_length_fraction`.
///
/// ```c
/// void LL_stack_set_length_fraction(double fraction) {
///     Signed length = (Signed)(MAX_STACK_SIZE * fraction);
///     Signed os_limit = _ll_stack_os_limit();
///     if (os_limit > 0) {
///         Signed cap = os_limit - (os_limit >> 2);
///         if (cap > 0 && length > cap)
///             length = cap;
///     }
///     rpy_stacktoobig.stack_length = length;
/// }
/// ```
///
/// `sys.setrecursionlimit()` scales `length` linearly, so a high limit
/// can push it past the real OS stack and segfault before the check
/// ever reports an overflow. Clamping to the stack the OS actually
/// gives this thread, minus a 25% margin for the frames between the
/// check and the guard page, keeps the check firing first.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_set_length_fraction(frac: f64) {
    let length = (MAX_STACK_SIZE as f64 * frac) as usize;
    REQUESTED_STACK_LENGTH.store(length, Ordering::Relaxed);
    PYRE_STACKTOOBIG
        .stack_length
        .store(effective_stack_length(), Ordering::Relaxed);
}

/// rpython/translator/c/src/stack.c:25-66 `LL_stack_too_big_slowpath`.
///
/// Four-case logic, line-by-line parity with the C source:
///
///   * `baseptr == NULL` — first time we see this thread, fall through
///     to the "update base to current" epilogue (stack.c:42-44).
///   * `0 <= diff <= max_stack_size` — within bounds, probably just had
///     a thread switch. Refresh the global cache from the TLS mirror
///     (stack.c:49) and return `0`.
///   * `-max_stack_size <= diff < 0` — stack underflowed; revise the
///     base upward via the shared "update base to current" epilogue
///     (stack.c:52-55), return `0`.
///   * else — real stack overflow (stack.c:56-58), return the current
///     thread's `report_error` mirror (which is cleared during critical-code
///     sections by [`pyre_stack_criticalcode_start`]).
///
/// The TLS mirror [`TL_STACK_END`] is the source of truth
/// (stack.c:40 `baseptr = tl1->stack_end;`); the global
/// `PYRE_STACKTOOBIG.stack_end` is a cache refreshed from it.
///
/// Returns `1` on real stack overflow (caller should raise
/// `RecursionError`), `0` otherwise.
#[unsafe(no_mangle)]
#[majit_macros::dont_look_inside]
pub extern "C" fn pyre_stack_too_big_slowpath(current: usize) -> u8 {
    let max_stack_size = effective_stack_length();
    PYRE_STACKTOOBIG
        .stack_length
        .store(max_stack_size, Ordering::Relaxed);
    // stack.c:38-40 OP_THREADLOCALREF_ADDR + baseptr = tl1->stack_end.
    // Explicit `usize` annotation pins the variable kind to Signed for
    // the pyre annotator; `Cell::get()` through a `thread_local!`
    // `.with` closure isn't resolved by the front-end, which would
    // otherwise default the binding to the GcRef Unknown sentinel
    // (`jtransform.rs`'s `ConcreteType::Unknown => 'r'` arms) and emit a
    // mixed-kind `int_ne/ri>i` at the `!= 0` compare below.
    let baseptr: usize = TL_STACK_END.with(|c| c.get());

    if baseptr != 0 {
        // stack.c:46 diff = baseptr - curptr  (signed). We model the
        // two signs using two unsigned wrapping subtractions — the one
        // that produces a small value indicates the actual sign.
        let diff = baseptr.wrapping_sub(current);
        if diff <= max_stack_size {
            // stack.c:47-50 within bounds, probably thread switch.
            // Refresh the global cache from the TLS source of truth.
            // TLS already holds `baseptr`, only the global cache needs
            // the store here (stack.c:49).
            PYRE_STACKTOOBIG.stack_end.store(baseptr, Ordering::Relaxed);
            return 0;
        }
        let neg_diff = current.wrapping_sub(baseptr);
        if neg_diff <= max_stack_size {
            // stack.c:52-55 stack underflow — fall through to the
            // "update base to current" epilogue below.
        } else {
            // stack.c:56-58 real stack overflow.
            return TL_REPORT_ERROR.with(|c| c.get());
        }
    }

    // stack.c:61-65 update the stack base pointer to the current
    // value. Write both the TLS mirror (source of truth) and the
    // global cache.
    TL_STACK_END.with(|c| c.set(current));
    PYRE_STACKTOOBIG.stack_end.store(current, Ordering::Relaxed);
    0
}

/// Backend-callable slowpath that wraps [`pyre_stack_too_big_slowpath`].
///
/// `rpython/rlib/rstack.py` `stack_check_slowpath` raises `_StackOverflow`.
/// The exception transformer stores that in ExcData (`llmodel.py`
/// `pos_exception` / `pos_exc_value`). The address registered with
/// `register_stack_check_addresses` is this function, and the assembler's
/// `_build_stack_check_slowpath` then jumps to `propagate_exception_path`,
/// which copies `pos_exc_value` into `jf_guard_exc`
/// (`compile.py` `PropagateExceptionDescr.handle_fail`).
///
/// On overflow, publish the `RecursionError` through
/// [`crate::runtime_ops::jit_publish_exception`] (`llmodel.py`
/// `_store_exception`), the same channel residual helpers use. That calls
/// the raiser `pyre-jit` registers with `register_jit_exc_raiser`, which
/// writes the backend `pos_exception` / `pos_exc_value` cells. It does not
/// park the object in [`TL_JIT_PENDING_EXCEPTION`].
///
/// Returns the same 0/1 that [`pyre_stack_too_big_slowpath`] returns,
/// so the backend can still branch on the result directly.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_check_slowpath_for_backend(current: usize) -> u8 {
    let r = pyre_stack_too_big_slowpath(current);
    if r != 0 {
        let exc_obj = w_exception_new(ExcKind::RecursionError, "maximum recursion depth exceeded");
        crate::runtime_ops::jit_publish_exception(exc_obj);
    }
    r
}

/// Cranelift-callable one-shot probe combining the rstack.py stack_check fast
/// path and the stack.c:25 slowpath into a single `extern "C"`
/// function. Cranelift's IR does not expose a "read current SP"
/// intrinsic, so it calls this helper instead of emitting the inline
/// `MOV [endaddr]; SUB sp; CMP [lengthaddr]` sequence the dynasm
/// backends use. Semantically equivalent to the inline probe + slow
/// path pair — just paid as one function call.
///
/// Returns `1` on real overflow (and [`pyre_stack_check_slowpath_for_backend`]
/// publishes a `RecursionError` into the backend exception cells), `0`
/// otherwise. The
/// return type is `i64` rather than `u8` so Cranelift's
/// `emit_host_call(..., Some(I64))` gets an unambiguous 64-bit
/// return value in the machine register (the System V AMD64 ABI does
/// not specify the high bytes of a narrower return).
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_check_for_jit_prologue() -> i64 {
    let current = current_sp();
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    let length = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    // rstack.py:58-60 fast path.
    let ofs = end.wrapping_sub(current);
    if ofs <= length {
        return 0;
    }
    pyre_stack_check_slowpath_for_backend(current) as i64
}

/// Drain the exception [`park_jit_pending_error`] parked, if any, and
/// convert it to `PyError`. The slot is thread-local, so another Rust
/// test thread cannot claim it. The prologue stack check does not write
/// this cell; its overflow is already in the backend `pos_exc_value`
/// (`llmodel.py` `_store_exception`) when `propagate_exception_path` runs.
#[inline]
#[majit_macros::dont_look_inside]
pub fn drain_jit_pending_exception() -> Result<(), PyError> {
    let obj = TL_JIT_PENDING_EXCEPTION.with(|slot| {
        let obj = slot.get();
        slot.set(0);
        obj
    });
    if obj != 0 {
        Err(unsafe { PyError::from_exc_object(obj as pyre_object::PyObjectRef) })
    } else {
        Ok(())
    }
}

/// One-word residual-call ABI for [`drain_jit_pending_exception`].
pub extern "C" fn drain_jit_pending_exception_jit_abi() -> i64 {
    match drain_jit_pending_exception() {
        Ok(()) => 0,
        Err(error) => crate::runtime_ops::jit_publish_residual_error(error),
    }
}

/// Peek whether an overflow exception is pending without draining it.
/// Used by tests and by codegen helpers that want to know whether to
/// skip work without claiming responsibility for surfacing it.
#[inline]
pub fn is_jit_overflow_pending() -> bool {
    TL_JIT_PENDING_EXCEPTION.with(|slot| slot.get() != 0)
}

/// Park `err` in this thread's pending slot so the next interpreter call
/// boundary re-raises it through [`drain_jit_pending_exception`].
///
/// `warmspot.py` re-raises an `ExitFrameWithExceptionRef` out of
/// `ll_portal_runner` straight into the portal's caller. `unpackiterable_driver`
/// is entered from a merge-point hook returning `()`, so the error travels
/// through this slot instead of a return value. Delivery is at the caller
/// loop's next dispatch, which for the drain is the `next(w_iterator)` that
/// immediately follows the hook — before `__next__` re-runs, so no side
/// effect repeats.
///
/// Overwrites any slot content. The prologue stack check is not a producer:
/// it publishes through `jit_publish_exception` into `pos_exception`.
pub fn park_jit_pending_error(mut err: PyError) {
    let obj = err.to_exc_object();
    if !obj.is_null() {
        set_jit_pending_exception(obj);
    }
}

/// Shadow-stack root walker for the pending-exception slot: a parked
/// `W_BaseException` is reachable from nothing else until the next call
/// boundary drains it, and the drain crosses collecting code
/// (`next()`/`__next__`). Registered once at JIT init next to the other
/// `register_extra_root_walker` slots, which reach only the *collecting*
/// thread's cell — every other mutator's cell is reached through the
/// per-mutator root area built from
/// [`capture_jit_pending_exception_area`].
pub fn walk_jit_pending_exception(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    TL_JIT_PENDING_EXCEPTION.with(|slot| unsafe {
        crate::eval::walk_raw_exception_cell_area(slot as *const _, visitor)
    });
}

/// Address of this thread's pending-exception cell, for registration in the
/// per-mutator `PyFrameRootArea` alongside the other thread-local exception
/// carriers. `rthread.py _trace_tlref` traces a GC-pointer thread
/// local by enumerating *every* thread's block
/// (`threadlocal.c:86-93 _RPython_ThreadLocals_Enum`); resolving the TLS on
/// whichever thread happened to start the collection would miss a stopped
/// mutator's parked exception.
pub(crate) fn capture_jit_pending_exception_area() -> *const std::cell::Cell<i64> {
    TL_JIT_PENDING_EXCEPTION.with(|slot| slot as *const _)
}

/// rpython/translator/c/src/stack.h:42 `LL_stack_criticalcode_start`.
/// Clears this thread's `report_error` mirror so its slowpath will not signal
/// a stack overflow during short critical-code sections.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_criticalcode_start() {
    TL_REPORT_ERROR.with(|c| c.set(0));
}

/// rpython/translator/c/src/stack.h:43 `LL_stack_criticalcode_stop`.
/// Re-enables this thread's `report_error` mirror so subsequent real
/// overflows raise `RecursionError` again.
#[unsafe(no_mangle)]
pub extern "C" fn pyre_stack_criticalcode_stop() {
    TL_REPORT_ERROR.with(|c| c.set(1));
}

/// Largest recursion limit ever requested on this process.  The native byte
/// budget is reserved from it, never from a lowered limit.
static NATIVE_STACK_RESERVE: AtomicI32 = AtomicI32::new(DEFAULT_RECURSION_LIMIT);

/// pypy/module/sys/vm.py `setrecursionlimit`, with Python 3.14's
/// logical-depth rejection before PyPy's byte-budget update.  The check must
/// use Python frame depth: native Rust frame size is unrelated to the Python
/// recursion budget and cannot decide whether a limit is below the current
/// interpreter depth.
pub fn set_recursion_limit(new_limit: i32) -> Result<(), PyError> {
    if new_limit <= 0 {
        return Err(PyError::value_error("recursion limit must be positive"));
    }
    // pypy/module/sys/vm.py:86-87 silent upper bound.
    let limit = new_limit.min(MAX_RECURSION_LIMIT);
    // Python/sysmodule.c `sys_setrecursionlimit_impl`: reject
    // when the current Python recursion depth has already reached the limit.
    let depth = crate::call::py_recursion_depth();
    if depth >= limit as u32 {
        return Err(PyError::recursion_error(format!(
            "cannot set the recursion limit to {limit} at the recursion depth {depth}: the limit is too low"
        )));
    }

    // Python 3.14 keeps Python recursion accounting separate from native stack
    // protection.  Preserve PyPy's native byte guard as a high-water
    // allocation: lowering the Python limit must not shrink it underneath the
    // already-running Rust interpreter stack.  Logical depth below is what
    // enforces the newly lowered Python limit.  The mark is the largest limit
    // ever requested, not the previous one — `setrecursionlimit(n)` called
    // twice would otherwise collapse the reservation from the startup 5000 to
    // `n` on the second call, and the native guard would then cut a recursion
    // short far above the Python limit.
    let reserved = NATIVE_STACK_RESERVE
        .fetch_max(limit, Ordering::Relaxed)
        .max(limit);
    pyre_stack_set_length_fraction(reserved as f64 * 0.001);
    crate::module::sys::state::set_recursion_limit(limit);
    // pypy/module/sys/vm.py:97 `increase_root_stack_depth(int(new_limit *
    // 0.001 * 163840))`.  Both of pyre's root stacks are sized off it: the
    // jitframe one compiled code pushes to, and the interpreter's `push_roots`
    // stack, which holds a slot per pinned livevar across a recursive call.
    // The allocation is eager (`shadowstack.py` resizes on the spot),
    // and the `MAX_RECURSION_LIMIT` clamp above is the only thing bounding it:
    // `vm.py:83-88` adds that same 10**6 ceiling precisely "because huge values
    // cause huge shadowstacks to be allocated (or MemoryErrors)".  A tighter
    // bound here would cap the root stack below the recursion limit it is
    // sized for.
    let root_stack_depth = (limit as f64 * 0.001 * 163840.0) as usize;
    majit_gc::shadow_stack::increase_root_stack_depth(root_stack_depth);
    pyre_object::gc_roots::increase_root_stack_depth(root_stack_depth);
    Ok(())
}

/// pypy/module/sys/vm.py `getrecursionlimit` parity. Reads
/// `space.sys.recursionlimit` via `module::sys::state`.
///
/// `#[dont_look_inside]`: reads the runtime-mutable `recursion_limit` global
/// (`sys.setrecursionlimit` mutates it), so the tracer residualises the read
/// rather than folding a build-time constant — the runtime-mutable-global
/// accessor pattern.
#[majit_macros::dont_look_inside]
pub fn get_recursion_limit() -> i32 {
    crate::module::sys::state::recursion_limit()
}

/// rpython/rlib/rstack.py `stack_check()` parity.
///
/// Fast path: `ofs = r_uint(end - current); if ofs <= r_uint(length):
/// return`. On miss, dispatch to [`pyre_stack_too_big_slowpath`] which
/// applies the 4-case logic (first-time capture, thread switch,
/// underflow revision, or real overflow).
///
/// `stack_check._always_inline_ = True` (`rstack.py`): the translator
/// inlines the body into the caller rather than residualizing a `CallN`.
/// The slow path stays opaque (`stack_check_slowpath._dont_inline_`).
#[inline]
#[majit_macros::always_inline]
pub fn stack_check() -> Result<(), PyError> {
    // CPython 3.14 checks `py_recursion_remaining`, i.e. Python interpreter
    // depth, independently of native stack protection.  pyre's matching
    // counter is bumped around every user-function call.
    check_recursion_depth()?;
    let current = current_sp();
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    let length = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    // rstack.py:58-60 ofs = r_uint(end - current); if ofs <= r_uint(length): return
    let ofs = end.wrapping_sub(current);
    if ofs <= length {
        return Ok(());
    }
    // rstack.py stack_check_slowpath(current)
    if pyre_stack_too_big_slowpath(current) != 0 {
        return Err(PyError::recursion_error("maximum recursion depth exceeded"));
    }
    Ok(())
}

/// One-word residual-call ABI for [`stack_check`].
pub extern "C" fn stack_check_jit_abi() -> i64 {
    match stack_check() {
        Ok(()) => 0,
        Err(error) => crate::runtime_ops::jit_publish_residual_error(error),
    }
}

/// Run only [`stack_check`]'s logical Python-activation-depth half.
///
/// A compiled fragment already carries the backend's native-stack probe at
/// its entry. Its guard exit still has to reproduce `PyFrame.execute_frame`'s
/// check against `sys.getrecursionlimit()`, without paying for that native
/// probe a second time when it enters the portal runner.
#[inline]
pub fn check_recursion_depth() -> Result<(), PyError> {
    recursion_depth_check(crate::call::py_recursion_depth())
}

/// The logical half of [`stack_check`]: `execute_frame`'s entry test against
/// `sys.getrecursionlimit()`.
///
/// Split out because the two halves have different homes once the code is
/// compiled.  The native half is what the backend emits at every fragment
/// entry (`_call_header_with_stack_check`), which is where upstream puts the
/// whole of `rstack.stack_check`; repeating it in a residual call at the same
/// boundary would only pay for it twice.  The logical half has no such home,
/// so it is the half a synthesized activation has to carry.
#[inline]
fn recursion_depth_check(depth: u32) -> Result<(), PyError> {
    if depth >= get_recursion_limit() as u32 {
        return Err(PyError::recursion_error("maximum recursion depth exceeded"));
    }
    Ok(())
}

/// rpython/rlib/rstack.py `stack_almost_full` parity.
///
/// ```python
/// def stack_almost_full():
///     current = llop.stack_current(Signed)
///     end = _stack_get_end()
///     length = 15 * (r_uint(_stack_get_length()) >> 4)
///     ofs = r_uint(end - current)
///     if ofs <= length:
///         return False    # fine
///     else:
///         _stack_too_big_slowpath(current)   # may update stack_end
///         end = _stack_get_end()
///         ofs = r_uint(end - current)
///         return ofs > length
/// ```
///
/// Returns `true` if the stack is more than 15/16ths full, computed
/// against the `sys.setrecursionlimit`-driven [`PYRE_STACKTOOBIG`]
/// budget rather than the OS-level thread stack. Callers use this to
/// gate bridge compilation (`compile.py:702-703`) and GC hooks
/// (`warmstate.py:430`). The slowpath is invoked when the cached
/// `stack_end` is stale so it can refresh, matching the two-step
/// check/slowpath/re-check pattern upstream.
// rstack.py `stack_almost_full._jit_look_inside_ = False` — the
// JIT never traces the budget arithmetic; calls residualize.
#[majit_macros::dont_look_inside]
pub extern "C" fn stack_almost_full() -> bool {
    // rstack.py:80 current = llop.stack_current(Signed)
    let current = current_sp();
    // rstack.py end = _stack_get_end()
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    // rstack.py length = 15 * (r_uint(_stack_get_length()) >> 4)
    let length_full = PYRE_STACKTOOBIG.stack_length.load(Ordering::Relaxed);
    let length = 15 * (length_full >> 4);
    // rstack.py:83-84 ofs = r_uint(end - current); if ofs <= length: return False
    let ofs = end.wrapping_sub(current);
    if ofs <= length {
        return false;
    }
    // rstack.py:86-89 slowpath may update stack_end, re-read and compare.
    let _ = pyre_stack_too_big_slowpath(current);
    let end = PYRE_STACKTOOBIG.stack_end.load(Ordering::Relaxed);
    let ofs = end.wrapping_sub(current);
    ofs > length
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reset every shared or thread-local stack-check slot back to its
    /// default — call from each test under `STACK_AND_JIT_TEST_LOCK`
    /// so neither another stack-state test nor a compiled trace can
    /// observe the synthetic values.
    fn reset_all() {
        REQUESTED_STACK_LENGTH.store(MAX_STACK_SIZE, Ordering::Relaxed);
        TL_OS_STACK_SIZE.with(|slot| slot.set(0));
        reset_stack_base();
        PYRE_STACKTOOBIG.report_error.store(1, Ordering::Relaxed);
        TL_REPORT_ERROR.with(|c| c.set(1));
        crate::module::sys::state::reset_recursion_limit_for_tests();
        NATIVE_STACK_RESERVE.store(DEFAULT_RECURSION_LIMIT, Ordering::Relaxed);
        clear_jit_pending_exception();
    }

    /// Plant a synthetic `stack_end` into both the TLS mirror
    /// (source of truth — stack.c:40) and the global cache
    /// (inline-probe visible — stack.c:14,49). Tests that exercise
    /// the slowpath must write both; the slowpath reads the TLS.
    fn plant_stack_end(value: usize) {
        TL_STACK_END.with(|c| c.set(value));
        PYRE_STACKTOOBIG.stack_end.store(value, Ordering::Relaxed);
    }

    /// Read this thread's captured stack base from the source of truth
    /// (`TL_STACK_END`, stack.c:40). The slowpath tests assert on this rather
    /// than the global cache `pyre_stack_get_end()`: the global is refreshed
    /// by any thread that runs `stack_check()` (stack.c:49,63), so in the
    /// multi-threaded test binary a concurrent interpreter test can overwrite
    /// it between the slowpath store and the read, making the global an
    /// unstable observation. The per-thread TLS mirror is race-free.
    fn captured_base() -> usize {
        TL_STACK_END.with(|c| c.get())
    }

    /// Acquire the test mutex.  `parking_lot` does not poison, so a single
    /// previous-test panic doesn't cascade into every follow-up test.
    fn lock_tests() -> parking_lot::MutexGuard<'static, ()> {
        test_support::stack_and_jit_test_guard()
    }

    #[test]
    fn fresh_thread_does_not_overflow() {
        let _g = lock_tests();
        reset_all();
        assert!(stack_check().is_ok());
    }

    #[test]
    fn shallow_check_does_not_overflow() {
        let _g = lock_tests();
        reset_all();
        for _ in 0..1000 {
            assert!(stack_check().is_ok());
        }
    }

    /// The requested byte budget, before [`effective_stack_length`]'s clamp.
    ///
    /// Budget assertions read this rather than `pyre_stack_get_length()`: the
    /// clamp is three quarters of whatever stack the running thread has, so on
    /// a host whose `RLIMIT_STACK` is smaller than the budget under test the
    /// stored length says nothing about what was requested. What the recursion
    /// limit controls is the request.
    fn requested_budget() -> usize {
        REQUESTED_STACK_LENGTH.load(Ordering::Relaxed)
    }

    #[test]
    fn default_recursion_limit_matches_python() {
        let _g = lock_tests();
        reset_all();
        assert_eq!(get_recursion_limit(), DEFAULT_RECURSION_LIMIT);
        assert_eq!(requested_budget(), MAX_STACK_SIZE);
        assert_eq!(pyre_stack_get_length(), effective_stack_length());
        reset_all();
    }

    #[test]
    fn set_recursion_limit_keeps_native_stack_budget_high_water_mark() {
        let _g = lock_tests();
        reset_all();
        // Lowering the Python logical limit must not shrink native protection
        // underneath the currently-running interpreter stack.
        set_recursion_limit(500).expect("500 is positive");
        assert_eq!(get_recursion_limit(), 500);
        assert_eq!(requested_budget(), MAX_STACK_SIZE);

        // Raising the limit grows both the logical limit and native budget.
        set_recursion_limit(2000).expect("2000 is positive");
        assert_eq!(get_recursion_limit(), 2000);
        assert_eq!(requested_budget(), 2 * MAX_STACK_SIZE);

        reset_all();
    }

    #[test]
    fn set_recursion_limit_rejects_non_positive() {
        let _g = lock_tests();
        reset_all();
        let err = set_recursion_limit(0).expect_err("0 is not positive");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        let err = set_recursion_limit(-5).expect_err("negative is not positive");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        // The limit must NOT have been mutated.
        assert_eq!(get_recursion_limit(), DEFAULT_RECURSION_LIMIT);
        reset_all();
    }

    #[test]
    fn set_recursion_limit_clamps_to_silent_upper_bound() {
        let _g = lock_tests();
        reset_all();
        // pypy/module/sys/vm.py:67 — values above 1_000_000 are clamped.
        set_recursion_limit(50_000_000).expect("positive");
        assert_eq!(get_recursion_limit(), MAX_RECURSION_LIMIT);
        reset_all();
    }

    #[test]
    fn configured_thread_stack_clamps_native_budget() {
        let _g = lock_tests();
        reset_all();
        configure_current_thread_stack_size(1024 * 1024);
        pyre_stack_set_length_fraction(20.0);
        assert_eq!(
            pyre_stack_get_length(),
            768 * 1024,
            "one quarter of the configured native stack must remain reserved"
        );
        reset_all();
    }

    #[test]
    fn configured_thread_stack_replaces_the_rlimit_fallback() {
        let _g = lock_tests();
        reset_all();
        // The clamp describes the stack the *running* thread was given.
        // `getrlimit(RLIMIT_STACK)` is only the fallback for a thread that
        // never announced one, and it reports the process's original thread —
        // typically 8 MiB, less than the budget a thread pyre spawns with a
        // large explicit stack is entitled to. Announcing the real size must
        // therefore be able to raise the clamp, not just lower it.
        //
        // Both stores are process-global, so keep the window to the two calls
        // and restore at once (see small_recursion_limit_triggers_overflow_sooner).
        configure_current_thread_stack_size(8 * MAX_STACK_SIZE);
        pyre_stack_set_length_fraction(4.0);
        let length = pyre_stack_get_length();
        reset_all();
        assert_eq!(
            length,
            4 * MAX_STACK_SIZE,
            "an announced stack far above RLIMIT_STACK must carry the full request"
        );
    }

    #[test]
    fn backend_slowpath_reports_overflow_without_pending_slot() {
        let _g = lock_tests();
        reset_all();
        // No `register_jit_exc_raiser` in this crate's unit tests, so the
        // slowpath cannot be observed through backend exception cells.
        // The contract here is the return value, and that the pending
        // slot — owned by `park_jit_pending_error` — stays empty.
        let above = current_sp().saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        let result = pyre_stack_check_slowpath_for_backend(current_sp());
        assert_eq!(result, 1, "slowpath must signal overflow");
        assert!(
            !is_jit_overflow_pending(),
            "slowpath must not park the overflow in the pending slot"
        );
        assert!(
            drain_jit_pending_exception().is_ok(),
            "pending slot stays empty"
        );
        reset_all();
    }

    #[test]
    fn backend_slowpath_leaves_pending_slot_empty_on_every_thread() {
        let _g = lock_tests();
        reset_all();

        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let child = std::thread::spawn(move || {
            reset_stack_base();
            clear_jit_pending_exception();
            let above = current_sp().saturating_add(2 * MAX_STACK_SIZE);
            plant_stack_end(above);
            assert_eq!(pyre_stack_check_slowpath_for_backend(current_sp()), 1);
            assert!(!is_jit_overflow_pending());
            assert!(drain_jit_pending_exception().is_ok());
            ready_tx.send(()).expect("notify main thread");
        });

        ready_rx.recv().expect("wait for child slowpath");
        assert!(
            !is_jit_overflow_pending(),
            "slowpath must not park an overflow visible to the main thread"
        );
        assert!(drain_jit_pending_exception().is_ok());
        child.join().expect("child thread must finish");
        reset_all();
    }

    #[test]
    fn jit_prologue_helper_returns_0_on_fast_path() {
        let _g = lock_tests();
        reset_all();
        // Default state: stack_end = current_sp() (via reset_all), so
        // the fast path sees a small positive offset and returns 0
        // without dispatching to the slowpath.
        assert_eq!(pyre_stack_check_for_jit_prologue(), 0i64);
        assert!(!is_jit_overflow_pending());
        reset_all();
    }

    #[test]
    fn small_recursion_limit_triggers_overflow_sooner() {
        let _g = lock_tests();
        reset_all();
        // A near base (MAX_STACK_SIZE / 100 above current) is within bounds at
        // the full budget but overflows once the budget shrinks to ~768 bytes
        // (recursionlimit 1, MAX_STACK_SIZE / 1000). stack_length is a process
        // global the interpreter reads on every frame and has no per-thread
        // mirror, so a tiny value left in it lets a concurrent interpreter frame
        // spuriously overflow; set_recursion_limit(1) would hold it tiny across a
        // wide window (its own stack_check + shadow-stack growth). Shrink the
        // budget only across the single slowpath classification and restore the
        // full budget at once. Assert through the TLS-authoritative slowpath, not
        // stack_check(), whose fast path also reads the shared global (see
        // captured_base).
        let above = current_sp().saturating_add(MAX_STACK_SIZE / 100);
        plant_stack_end(above);
        pyre_stack_set_length_fraction(0.001);
        let overflow = pyre_stack_too_big_slowpath(current_sp());
        pyre_stack_set_length_fraction(1.0);
        assert_eq!(
            overflow, 1,
            "a tiny budget must classify a near base as a real overflow"
        );
        reset_all();
    }

    #[test]
    fn slowpath_first_time_captures_base() {
        let _g = lock_tests();
        reset_all();
        // Simulate "never captured" state — stack.c:42-44 first-time path.
        plant_stack_end(0);
        let sp = current_sp();
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 0, "first-time capture must not signal overflow");
        let captured = captured_base();
        assert_eq!(captured, sp, "slowpath must update stack_end to current");
        reset_all();
    }

    #[test]
    fn slowpath_thread_switch_cache_hit() {
        let _g = lock_tests();
        reset_all();
        // baseptr very close to current — within max_stack_size, slowpath
        // should classify this as stack.c:47-50 "within bounds / thread
        // switch" and return 0 without mutating stack_end.
        let sp = current_sp();
        let base = sp.saturating_add(4096);
        plant_stack_end(base);
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 0);
        assert_eq!(
            captured_base(),
            base,
            "within-bounds hit must not mutate base"
        );
        reset_all();
    }

    #[test]
    fn slowpath_underflow_revises_base_upward() {
        let _g = lock_tests();
        reset_all();
        // baseptr BELOW current within max_stack_size — stack.c:52-55
        // underflow case. Slowpath should revise base upward to current
        // and return 0.
        let sp = current_sp();
        let below = sp.wrapping_sub(4096);
        plant_stack_end(below);
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 0);
        assert_eq!(captured_base(), sp, "underflow must revise base to current");
        reset_all();
    }

    #[test]
    fn slowpath_real_overflow_returns_report_error() {
        let _g = lock_tests();
        reset_all();
        let sp = current_sp();
        let above = sp.saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(
            overflow, 1,
            "real overflow must signal with report_error=true"
        );
        reset_all();
    }

    #[test]
    fn slowpath_criticalcode_suppresses_overflow() {
        let _g = lock_tests();
        reset_all();
        let sp = current_sp();
        let above = sp.saturating_add(2 * MAX_STACK_SIZE);
        plant_stack_end(above);
        pyre_stack_criticalcode_start();
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(
            overflow, 0,
            "criticalcode must suppress overflow signalling"
        );
        pyre_stack_criticalcode_stop();
        let overflow = pyre_stack_too_big_slowpath(sp);
        assert_eq!(overflow, 1, "stop must re-enable overflow signalling");
        reset_all();
    }

    #[test]
    fn criticalcode_suppression_is_thread_local() {
        let _g = lock_tests();
        reset_all();
        let entered = std::sync::Arc::new(std::sync::Barrier::new(2));
        let release = std::sync::Arc::new(std::sync::Barrier::new(2));
        let worker_entered = entered.clone();
        let worker_release = release.clone();
        let worker = std::thread::spawn(move || {
            pyre_stack_criticalcode_start();
            worker_entered.wait();
            worker_release.wait();
            pyre_stack_criticalcode_stop();
        });
        entered.wait();

        let sp = current_sp();
        plant_stack_end(sp.saturating_add(2 * MAX_STACK_SIZE));
        assert_eq!(
            pyre_stack_too_big_slowpath(sp),
            1,
            "another thread's criticalcode must not suppress this thread"
        );

        release.wait();
        worker.join().unwrap();
        reset_all();
    }

    #[test]
    fn end_and_length_addresses_are_stable() {
        let _g = lock_tests();
        // rstack.py `_stack_get_end_adr` / `_stack_get_length_adr`
        // parity: addresses must be stable across calls (so the JIT
        // backend can embed them as imm64 operands).
        let end_adr_1 = pyre_stack_get_end_adr();
        let end_adr_2 = pyre_stack_get_end_adr();
        assert_eq!(
            end_adr_1, end_adr_2,
            "pyre_stack_get_end_adr must return a stable address"
        );
        let length_adr_1 = pyre_stack_get_length_adr();
        let length_adr_2 = pyre_stack_get_length_adr();
        assert_eq!(
            length_adr_1, length_adr_2,
            "pyre_stack_get_length_adr must return a stable address"
        );
        assert_ne!(
            end_adr_1, length_adr_1,
            "end and length must live at distinct addresses"
        );
    }

    #[test]
    fn end_and_length_addresses_reflect_layout() {
        let _g = lock_tests();
        // stack.h:23-27 `rpy_stacktoobig_t` layout: stack_end at
        // offset 0, stack_length at offset sizeof(usize). Verify pyre's
        // PyreStackTooBig matches.
        let end_adr = pyre_stack_get_end_adr();
        let length_adr = pyre_stack_get_length_adr();
        let struct_adr = &raw const PYRE_STACKTOOBIG as usize;
        assert_eq!(end_adr, struct_adr, "stack_end at offset 0");
        assert_eq!(
            length_adr,
            struct_adr + std::mem::size_of::<usize>(),
            "stack_length at offset sizeof(usize)"
        );
    }

    #[test]
    fn addresses_observe_stored_values() {
        let _g = lock_tests();
        reset_all();
        // Store via FFI, read via raw pointer — confirms the atomic
        // storage is bit-compatible with a plain usize load.
        pyre_stack_set_length_fraction(0.5);
        // What the fraction scales is the request; what lands in the word is
        // that request after the OS clamp, which depends on this host's
        // RLIMIT_STACK. Assert the two separately so neither reading needs a
        // host with a stack larger than the budget under test.
        assert_eq!(requested_budget(), MAX_STACK_SIZE / 2);
        let length_adr = pyre_stack_get_length_adr();
        let loaded = unsafe { *(length_adr as *const usize) };
        assert_eq!(
            loaded,
            effective_stack_length(),
            "raw load through adr must observe the FFI store"
        );
        reset_all();
    }

    #[test]
    fn slowpath_capture_survives_concurrent_global_refresh() {
        let _g = lock_tests();
        reset_all();
        // A first-time capture writes both the TLS source of truth and the
        // global cache. Another thread running stack_check() may then refresh
        // the shared global cache to its own base (stack.c:49,63); model that
        // with a direct store. The per-thread TLS mirror is unaffected — which
        // is why the slowpath tests read it (via captured_base) rather than the
        // racy global cache pyre_stack_get_end().
        plant_stack_end(0);
        let sp = current_sp();
        assert_eq!(pyre_stack_too_big_slowpath(sp), 0);
        assert_eq!(captured_base(), sp);
        let foreign = sp.wrapping_add(MAX_STACK_SIZE / 2);
        PYRE_STACKTOOBIG.stack_end.store(foreign, Ordering::Relaxed);
        assert_eq!(
            captured_base(),
            sp,
            "a concurrent global-cache refresh must not perturb this thread's captured base"
        );
        // The global cache itself is not asserted here: it is process-wide and
        // any other test thread running interpreter code refreshes it via
        // pyre_stack_too_big_slowpath first-time capture, so its value is racy
        // by construction — which is exactly why the invariant is read from the
        // TLS mirror (captured_base) above.
        reset_all();
    }
}
