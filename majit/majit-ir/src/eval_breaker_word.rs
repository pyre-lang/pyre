//! The single process-global eval-breaker word polled by JIT-compiled loop
//! back-edges. One `AtomicUsize` whose bits fold the two former back-edge
//! polls into one load + one nonzero branch:
//!   bit0 EB_GC_INTERP — process-stable `PYRE_GC_INTERP` dispatch gate.  It
//!                       shares the word to spare the interpreter a second
//!                       per-opcode atomic load, but it is not itself a reason
//!                       to leave machine code.  It occupies the lowest bit so
//!                       that every breaker bit outranks it numerically and a
//!                       compiled poll can separate the two with one unsigned
//!                       compare against `JIT_BREAKER_FLOOR` — see that
//!                       constant.
//!   bit1 EB_ASYNC — an async ticker request; OR'd in by the OS signal handler
//!                   and action dispatcher, then copied into
//!                   `pypysig_counter` at a safe interpreter checkpoint.
//!   bit2 EB_STW   — mirrors `GC_SYNC.stw_requested`; OR'd in by the collector
//!                   while it drains mutators to safepoints.
//!   bit3 EB_FINALIZING — mirrors interpreter finalization; once armed,
//!                        non-owner mutators park before their next opcode.
//!   bit4 EB_GC    — the old-gen allocator reached the next-major threshold;
//!                   OR'd in by the collector, consumed by the interpreter
//!                   dispatch-loop GC safepoint.
//!   bit5 EB_MEMORY_ERROR — a bounded major collection exhausted the heap;
//!                   OR'd in by the collector where upstream raises, consumed
//!                   by the dispatch loop, which raises `MemoryError` there.
//!                   Published for every thread because it is a deopt trigger,
//!                   but delivered only to the thread whose collection
//!                   exhausted the heap; see `set_memory_error`.
//! A compiled loop loads the whole word at the back-edge and deopts to the
//! interpreter when it is non-zero. The interpreter/warm-up loop and the STW
//! park gate remain authoritative; this word is only the JIT's deopt trigger.
//!
//! Immortal process-global, zero from process start, so the guard is harmless
//! until a bit is armed. The trace records the address as the poll's constant
//! operand; the published-address holder (`EVAL_BREAKER_WORD_ADDR`) reads `0`
//! until published, and no poll is recorded in that case.
//!
//! The STW bit and authoritative request occupy two locations. A relaxed poll
//! can briefly deopt before the request is visible, then resume and re-deopt
//! until coherence propagates it; this is a bounded park-latency window.

use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering};

/// bit0 — interpreter-path allocation/collection integration is enabled.
/// Lowest bit so that it is the only value below `JIT_BREAKER_FLOOR`.
pub const EB_GC_INTERP: usize = 1;
/// bit1 — async action / signal pending (mirrors a negative ticker).
pub const EB_ASYNC: usize = 2;
/// bit2 — GC stop-the-world requested (mirrors `GC_SYNC.stw_requested`).
pub const EB_STW: usize = 4;
/// bit3 — interpreter finalization has begun (terminal, never cleared).
pub const EB_FINALIZING: usize = 8;
/// bit4 — the old-gen allocator crossed the next-major threshold and a
/// collection is owed at the next root-complete point.
pub const EB_GC: usize = 16;
/// bit5 — a bounded major collection reached `max_heap_size` and a
/// `MemoryError` is owed at the next root-complete point.
pub const EB_MEMORY_ERROR: usize = 32;
/// Bits that require a compiled loop to deopt to the interpreter.
///
/// `EB_GC` belongs here: the allocator only arms the request, and the
/// collection itself runs at the dispatch-loop safepoint, so a compiled loop
/// has to leave machine code for the request to be serviced at all.
///
/// `EB_MEMORY_ERROR` belongs here for the same reason: the exception is raised
/// by the dispatch loop, so a compiled loop that never returns to it would run
/// on past a heap the collector has already declared exhausted.
pub const JIT_BREAKER_MASK: usize = EB_ASYNC | EB_STW | EB_FINALIZING | EB_GC | EB_MEMORY_ERROR;

/// Every bit the word can carry, breaker or not.
pub const ALL_EVAL_BREAKER_BITS: usize =
    EB_GC_INTERP | EB_ASYNC | EB_STW | EB_FINALIZING | EB_GC | EB_MEMORY_ERROR;

/// The lowest breaker bit. Every bit in `JIT_BREAKER_MASK` is at or above it
/// and every other bit is below it, so `word >= JIT_BREAKER_FLOOR` decides
/// "some breaker is armed" for the whole word. That lets a compiled back-edge
/// poll test the word with a single unsigned compare instead of masking and
/// then testing the remainder, while the interpreter keeps reading its own
/// bits out of the same load.
pub const JIT_BREAKER_FLOOR: usize = EB_ASYNC;

// The compare above is only equivalent to the mask while the two bit groups
// stay separated by `JIT_BREAKER_FLOOR`. Adding a non-breaker bit above the
// floor, or a breaker bit below it, would silently turn every back-edge poll
// into the wrong answer, so state the partition as a compile-time fact.
const _: () = {
    assert!(JIT_BREAKER_MASK & (JIT_BREAKER_FLOOR - 1) == 0);
    assert!(ALL_EVAL_BREAKER_BITS & !JIT_BREAKER_MASK & !(JIT_BREAKER_FLOOR - 1) == 0);
};

/// The shared eval-breaker word (see module docs).
static EVAL_BREAKER_WORD: AtomicUsize = AtomicUsize::new(0);

/// Address of the action ticker (`signals.c pypysig_counter`). Null until
/// the process flag publishes it. Armers store `-1` through this pointer so
/// the next `decrement_ticker` takes the `action_dispatcher` path. A fire
/// that arrives first sticks in `TICKER_FORCED` and is folded in at publish.
static ACTION_TICKER: AtomicPtr<AtomicIsize> = AtomicPtr::new(std::ptr::null_mut());
static TICKER_FORCED: AtomicIsize = AtomicIsize::new(0);

/// Address of the action ticker (`signals.c pypysig_counter`), or 0 before
/// the process action flag has published it.  Compiled back-edges embed this
/// stable address and perform PyPy's raw ticker load/store sequence.
pub fn action_ticker_addr() -> usize {
    ACTION_TICKER.load(Ordering::Acquire) as usize
}

/// Width used by `raw_load_i` / `raw_store_i` for the action ticker.
pub const ACTION_TICKER_WORD_SIZE: usize = size_of::<AtomicIsize>();

/// Publish the `pypysig_counter` ticker cell. The cell is an `AtomicIsize`;
/// signal handlers and collectors store `-1` into it.
pub fn publish_action_ticker(ptr: *mut AtomicIsize) {
    if ptr.is_null() {
        return;
    }
    // Publish before draining `TICKER_FORCED`: a fire that found no cell is
    // picked up by the swap, and every later one writes the cell itself.
    ACTION_TICKER.store(ptr, Ordering::Release);
    if TICKER_FORCED.swap(0, Ordering::AcqRel) < 0 {
        unsafe { (*ptr).store(-1, Ordering::Relaxed) };
    }
}

/// `AbstractActionFlag.fire` / `signals.c pypysig_pushback`: force the next
/// `decrement_ticker` below zero. Lock-free; safe in an OS signal handler
/// once the ticker cell is an atomic.
pub fn fire_action_ticker() {
    let ptr = ACTION_TICKER.load(Ordering::Relaxed);
    if ptr.is_null() {
        TICKER_FORCED.store(-1, Ordering::Relaxed);
        return;
    }
    unsafe { (*ptr).store(-1, Ordering::Relaxed) };
}

/// Width of the word, in bytes. The back-edge poll's load descriptor must use
/// exactly this size: a wider load reads past the word into the adjacent
/// static, so the poll's nonzero test would always be true and every back-edge
/// guard would fail. Pointer-width rather than fixed-64 keeps `fetch_or`
/// lock-free — `set_async` runs inside an OS signal handler.
pub const EVAL_BREAKER_WORD_SIZE: usize = size_of::<AtomicUsize>();

/// Published address of `EVAL_BREAKER_WORD`; `0` until published.
static EVAL_BREAKER_WORD_ADDR: AtomicUsize = AtomicUsize::new(0);

/// Publish the word's address for the tracer to record. Idempotent — the
/// address is an immortal static, so re-publishing stores the same value.
pub fn publish_addr() {
    EVAL_BREAKER_WORD_ADDR.store(
        &EVAL_BREAKER_WORD as *const AtomicUsize as usize,
        Ordering::Relaxed,
    );
}

/// Address the trace records as the back-edge poll's constant, or `0` if not
/// published — in which case no poll is recorded.
pub fn eval_breaker_word_addr() -> usize {
    EVAL_BREAKER_WORD_ADDR.load(Ordering::Relaxed)
}

#[inline]
pub fn load() -> usize {
    EVAL_BREAKER_WORD.load(Ordering::Relaxed)
}

/// One-word residual-call ABI for [`load`].
///
/// A value-returning residual lowers to a direct `call_indirect` typed
/// `() -> i64`, and `usize` is narrower than a word on wasm32, where the
/// call type-checks its callee.
pub extern "C" fn load_jit_abi() -> i64 {
    load() as i64
}

// async (bit0): armed by the OS signal handler / action dispatcher
// `fetch_or` is a single lock-free atomic RMW → async-signal-safe.
pub fn set_async() {
    EVAL_BREAKER_WORD.fetch_or(EB_ASYNC, Ordering::Relaxed);
    // `signals.c pypysig_pushback` drives `pypysig_counter` negative in the
    // handler. The breaker bit stays the compiled back-edge deopt; the ticker
    // is what the interpreter's `decrement_ticker` observes.
    fire_action_ticker();
}
pub fn clear_async() {
    EVAL_BREAKER_WORD.fetch_and(!EB_ASYNC, Ordering::Relaxed);
}

// stw (bit1): armed/cleared by the collector under the quiesce lock
pub fn set_stw() {
    EVAL_BREAKER_WORD.fetch_or(EB_STW, Ordering::Release);
    fire_action_ticker();
}
pub fn clear_stw() {
    EVAL_BREAKER_WORD.fetch_and(!EB_STW, Ordering::Release);
}

pub fn set_finalizing() {
    EVAL_BREAKER_WORD.fetch_or(EB_FINALIZING, Ordering::Release);
    fire_action_ticker();
}

pub fn set_gc_interp() {
    EVAL_BREAKER_WORD.fetch_or(EB_GC_INTERP, Ordering::Release);
}

// gc (bit4): armed by the old-gen allocator, consumed by the safepoint

/// Request a major collection at the next root-complete point.
///
/// `external_malloc` (incminimark.py) tests `threshold_reached` in the
/// allocator and drives `minor_collection_with_major_progress` there. pyre
/// cannot collect at an arbitrary allocation site — a Rust caller's locals are
/// not roots — so the check stays in the allocator and only the action is
/// deferred: this bit fails the next back-edge poll, and the interpreter
/// dispatch loop, where the frame walker sees the whole root set, performs the
/// collection.
pub fn set_gc() {
    EVAL_BREAKER_WORD.fetch_or(EB_GC, Ordering::Relaxed);
    // `finish_alloc_in_oldgen` arms this when the threshold is already
    // reached. Firing the ticker makes `action_dispatcher` run on the next
    // bytecode instead of waiting out the periodic poll.
    fire_action_ticker();
}

/// Consume the pending-collection request, reporting whether one was armed.
///
/// The caller must clear unconditionally — before any decision about whether it
/// can service the request. A bit left armed by a safepoint that declined to
/// collect fails every subsequent back edge, so a compiled loop would deopt on
/// each iteration instead of once.
pub fn take_gc() -> bool {
    // The taker runs on every eligible bytecode dispatch, the armer only when
    // the threshold is crossed; read before the read-modify-write so the
    // common case is a plain load.
    if EVAL_BREAKER_WORD.load(Ordering::Relaxed) & EB_GC == 0 {
        return false;
    }
    EVAL_BREAKER_WORD.fetch_and(!EB_GC, Ordering::Relaxed) & EB_GC != 0
}

// memory error (bit5): armed by the collector, raised by the dispatch loop

thread_local! {
    /// Whether *this* thread armed a `MemoryError` its own dispatch loop has
    /// not raised yet.
    ///
    /// The exception belongs to the thread whose collection exhausted the
    /// heap, the way upstream's `raise MemoryError` unwinds through whichever
    /// driver was on the stack. The bit alone cannot say that: it is process
    /// global, and a stop-the-world collection resumes the other mutators
    /// before the collecting one returns to its dispatch loop, so without this
    /// an unrelated thread could reach a back edge first and take an exception
    /// nothing it did earned — leaving the thread that did exhaust the heap
    /// running on toward the fatal rung.
    static MEMORY_ERROR_OWED: OwedMemoryError = const { OwedMemoryError(Cell::new(false)) };
}

/// This thread's half of the debt [`MEMORY_ERROR_OWERS`] counts.
///
/// A type with a destructor rather than a bare `Cell`, because a thread can
/// exit still owing one: the exception is raised by a dispatch loop, and a
/// thread whose loop has already returned never reaches another. Left on the
/// census that debt is unpayable — the count never falls to zero, so no later
/// owner clears the bit, and `EB_MEMORY_ERROR` is in `JIT_BREAKER_MASK`, so
/// every back edge in the process fails from then on. It also freezes the
/// collector's ladder, which reads [`memory_error_armed`] to tell a breach
/// that repeats an exception the program already has from one that is new.
struct OwedMemoryError(Cell<bool>);

impl Drop for OwedMemoryError {
    fn drop(&mut self) {
        if self.0.replace(false) {
            release_one_owed();
        }
    }
}

/// Take one undelivered exception off the census, clearing the bit if it was
/// the last.
fn release_one_owed() {
    if MEMORY_ERROR_OWERS.fetch_sub(1, Ordering::AcqRel) != 1 {
        return;
    }
    EVAL_BREAKER_WORD.fetch_and(!EB_MEMORY_ERROR, Ordering::Relaxed);
    // Another thread may have armed one between the decrement and the clear,
    // and set a bit this then removed. Re-publish the summary it is owed;
    // setting an already-set bit is what the racing arm did anyway, so this is
    // idempotent rather than a second signal.
    if MEMORY_ERROR_OWERS.load(Ordering::Acquire) != 0 {
        EVAL_BREAKER_WORD.fetch_or(EB_MEMORY_ERROR, Ordering::Relaxed);
        fire_action_ticker();
    }
}

/// Rebuild the census around the one thread `fork()` leaves running.
///
/// `rthread.thread_after_fork()` parity: the child has only the thread that
/// called `fork()`, and the others' debts have no dispatch loop left to pay
/// them there. A vanished thread runs no destructor, so [`OwedMemoryError`]
/// cannot clear them either, and keeping the count would arm the bit in the
/// child for good.
pub fn memory_error_after_fork_child() {
    let owed_here = MEMORY_ERROR_OWED.with(|owed| owed.0.get());
    MEMORY_ERROR_OWERS.store(usize::from(owed_here), Ordering::SeqCst);
    if owed_here {
        EVAL_BREAKER_WORD.fetch_or(EB_MEMORY_ERROR, Ordering::Relaxed);
        fire_action_ticker();
    } else {
        EVAL_BREAKER_WORD.fetch_and(!EB_MEMORY_ERROR, Ordering::Relaxed);
    }
}

/// How many threads owe an undelivered `MemoryError`.
///
/// The bit is the summary a back edge polls, and only the last owner to
/// deliver may clear it. Counting is what tells that owner apart from the
/// others.
static MEMORY_ERROR_OWERS: AtomicUsize = AtomicUsize::new(0);

/// Record that a bounded major collection reached `max_heap_size`.
///
/// incminimark.py `major_collection_step` reacts to that with a plain `raise
/// MemoryError`, so upstream's exception surfaces wherever the collection was
/// driven from. Most of pyre's collections are driven from the dispatch-loop
/// safepoint, which returns `()` and cannot raise, so the collector arms this
/// bit and the loop raises on the next dispatch instead. The bit rather than a
/// return value, because the safepoint is one of several drivers — an
/// allocation, a finalizer run and an explicit collection request reach the
/// same collection — and only the dispatch loop can turn any of them into an
/// exception.
///
/// Call this on the thread that drove the collection: the exception is owed to
/// that thread, and the bit only publishes that one is owed to someone.
pub fn set_memory_error() {
    MEMORY_ERROR_OWED.with(|owed| {
        if owed.0.replace(true) {
            // Already owed here and not yet delivered, so the count and the
            // bit already stand for this thread.
            return;
        }
        MEMORY_ERROR_OWERS.fetch_add(1, Ordering::AcqRel);
        EVAL_BREAKER_WORD.fetch_or(EB_MEMORY_ERROR, Ordering::Relaxed);
        fire_action_ticker();
    });
}

/// Whether a `MemoryError` armed by [`set_memory_error`] is still owed, to any
/// thread.
///
/// [`take_memory_error`] clears the bit as the last owner delivers, so this
/// reads true for exactly the window between arming an exception and every
/// thread that was owed one raising it. The collector reads it to tell a breach
/// that arrives inside that window from one that arrives after the program has
/// had its exception — a question about the program, not about a thread, which
/// is why this reads the process-wide summary.
pub fn memory_error_armed() -> bool {
    EVAL_BREAKER_WORD.load(Ordering::Relaxed) & EB_MEMORY_ERROR != 0
}

/// Whether *this* thread is the one still owed a `MemoryError`.
///
/// The collector's max-heap ladder asks it to tell a breach that repeats an
/// exception this thread has already been given from one that is new to it.
/// [`memory_error_armed`] cannot answer that: it reports the process-wide
/// summary, so an exception owed to another thread would read as this thread's
/// own and silence a breach that thread never caused.
pub fn memory_error_owed_here() -> bool {
    MEMORY_ERROR_OWED.with(|owed| owed.0.get())
}

/// Consume the `MemoryError` owed to *this* thread, reporting whether one was.
///
/// A thread that is not owed one leaves the bit alone and deopts again at its
/// next back edge, until the owner delivers. That window is bounded the way the
/// STW one is: arming happens inside the owner's own safepoint, so its very
/// next dispatch takes it.
pub fn take_memory_error() -> bool {
    // Same shape as `take_gc`: the taker runs per dispatch and the armer only
    // when a bounded heap is exhausted, so keep the common case a plain load.
    if EVAL_BREAKER_WORD.load(Ordering::Relaxed) & EB_MEMORY_ERROR == 0 {
        return false;
    }
    MEMORY_ERROR_OWED.with(|owed| {
        if !owed.0.replace(false) {
            return false;
        }
        release_one_owed();
        true
    })
}

/// One-word residual-call ABI for [`take_memory_error`].
pub extern "C" fn take_memory_error_jit_abi() -> i64 {
    take_memory_error() as i64
}

/// Depth of the operation chain between the poll's load and its guard.
///
/// The recorder emits `RawLoadI -> UintGe -> GuardFalse`, so one link
/// separates the guard's condition from the load. The walk below allows
/// a few more so that an optimizer pass inserting or splitting one link does
/// not silently stop matching, and stops well before a long pure chain.
const POLL_CHAIN_DEPTH: usize = 6;

/// Whether `guard`'s condition is a back-edge poll of this word.
///
/// The poll is recorded as
/// `GuardFalse(UintGe(RawLoadI(addr, 0), JIT_BREAKER_FLOOR))`. The
/// match anchors on the `RawLoadI` of the published address rather than on the
/// whole shape: that load is the one link the recorder cannot drop (it must
/// stay non-pure and outside the always-pure range, or CSE forwards the
/// preamble's guarded-zero value into the loop body), so walking back to it
/// survives rewrites of the links in between.
///
/// An unpublished address reads `0` and no poll is recorded, so the walk
/// declines rather than matching an unrelated load from a null constant.
pub fn is_back_edge_poll_guard(guard: &crate::resoperation::Op) -> bool {
    let addr = eval_breaker_word_addr();
    if addr == 0 {
        return false;
    }
    // Control-flow guards (`GUARD_NOT_FORCED`, `GUARD_NO_EXCEPTION`, ...) carry
    // no condition operand at all.
    if guard.num_args() == 0 {
        return false;
    }
    let mut operand = guard.arg(0);
    for _ in 0..POLL_CHAIN_DEPTH {
        let Some(op) = operand.bound_op() else {
            return false;
        };
        if op.num_args() == 0 {
            return false;
        }
        if op.opcode == crate::resoperation::OpCode::RawLoadI {
            return op.arg(0).const_int() == Some(addr as i64);
        }
        operand = op.arg(0);
    }
    false
}

/// Every flag must fit in the word the poll actually loads. Checked per target,
/// so a flag too wide for a 32-bit `usize` fails the wasm32 build rather than
/// silently reading as unarmed there.
const _: () = assert!(ALL_EVAL_BREAKER_BITS < (1 << (EVAL_BREAKER_WORD_SIZE * 8 - 1)));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resoperation::{Op, OpCode};
    use crate::value::Const;

    fn bind(op: Op) -> crate::operand::Operand {
        crate::operand::Operand::from_bound_op(&crate::resoperation::OpRc::new(op))
    }

    /// Both memory-error tests read the process-global bit and assert on its
    /// resting state, so they cannot run at the same time as each other.
    static MEMORY_ERROR_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// The owed `MemoryError` is one-shot: armed once by the collector, raised
    /// once by the dispatch loop of the thread it is owed to. A second take
    /// must see nothing, or one breach would raise two `MemoryError`s.
    ///
    /// It must also be a bit the back-edge poll tests, since the raise happens
    /// in the dispatch loop and a compiled loop has to leave machine code to
    /// get there.
    #[test]
    fn the_owed_memory_error_is_taken_exactly_once() {
        let _guard = MEMORY_ERROR_TEST_LOCK.lock();
        assert!(
            !take_memory_error(),
            "nothing has armed the bit, so it must not report one pending"
        );
        set_memory_error();
        assert_ne!(
            load() & JIT_BREAKER_MASK,
            0,
            "an armed MemoryError must fail the back-edge poll"
        );
        assert!(take_memory_error(), "the armed bit is reported once");
        assert!(!take_memory_error(), "and only once");
        assert_eq!(
            load() & EB_MEMORY_ERROR,
            0,
            "taking it clears it, so later back edges are not deopted"
        );
    }

    /// The exception belongs to the thread whose collection exhausted the
    /// heap, not to whichever dispatch loop polls first.
    ///
    /// Upstream raises inside the collection, so the driver on that thread's
    /// stack is the one that receives it. Pyre defers through a process-global
    /// word, and a stop-the-world collection resumes the other mutators before
    /// the collecting one returns to its own dispatch loop — so without
    /// ownership the first unrelated thread to reach a back edge would take an
    /// exception nothing it did earned, and the thread that did exhaust the
    /// heap would run on toward the fatal rung with nothing owed to it.
    #[test]
    fn the_owed_memory_error_goes_to_the_thread_that_armed_it() {
        let _guard = MEMORY_ERROR_TEST_LOCK.lock();
        // The owner stays alive until it has delivered. A thread that exits
        // still owing hands the debt back — see the test below — so joining it
        // here would clear the very bit this test is watching.
        let (armed_tx, armed_rx) = std::sync::mpsc::channel();
        let (deliver_tx, deliver_rx) = std::sync::mpsc::channel();
        let owner = std::thread::spawn(move || {
            set_memory_error();
            // The summary is published for everyone, so a compiled loop on any
            // thread leaves machine code — the bit is the deopt trigger, not
            // the delivery.
            assert_ne!(load() & EB_MEMORY_ERROR, 0);
            assert!(memory_error_armed());
            armed_tx.send(()).unwrap();
            deliver_rx.recv().unwrap();
            assert!(take_memory_error(), "the owner is the thread that raises");
        });
        armed_rx.recv().unwrap();

        assert!(
            memory_error_armed(),
            "the exception is still owed, so the summary stays published"
        );
        assert!(
            !take_memory_error(),
            "this thread armed nothing, so it is owed nothing"
        );
        assert_ne!(
            load() & EB_MEMORY_ERROR,
            0,
            "and a thread that is owed nothing must not clear the owner's bit"
        );

        deliver_tx.send(()).unwrap();
        owner.join().unwrap();
        assert_eq!(
            load() & EB_MEMORY_ERROR,
            0,
            "the last owner to deliver clears the summary"
        );
    }

    /// A thread can exit while still owed one, and the debt has to leave the
    /// census with it. The exception is raised by a dispatch loop, so a thread
    /// whose loop has already returned never reaches another; left counted,
    /// nothing can ever clear the bit again.
    #[test]
    fn a_thread_that_exits_still_owing_hands_the_debt_back() {
        let _guard = MEMORY_ERROR_TEST_LOCK.lock();
        std::thread::spawn(|| {
            set_memory_error();
            assert!(
                memory_error_armed(),
                "the summary is published while it lives"
            );
        })
        .join()
        .unwrap();

        assert_eq!(
            load() & EB_MEMORY_ERROR,
            0,
            "the owner exited without delivering, so nothing is owed any more"
        );
        assert!(
            !take_memory_error(),
            "and no other thread inherits an exception it never earned"
        );
    }

    /// `fork()` leaves one thread running, so in the child every other thread's
    /// debt is unpayable. The census has to be rebuilt around the survivor.
    #[test]
    fn the_fork_child_keeps_only_the_surviving_threads_debt() {
        let _guard = MEMORY_ERROR_TEST_LOCK.lock();
        // A sibling's debt, modelled by hand: a thread that vanishes at
        // `fork()` runs no destructor, so an owner spawned here — which does —
        // cannot stand in for one.
        MEMORY_ERROR_OWERS.fetch_add(1, Ordering::AcqRel);
        EVAL_BREAKER_WORD.fetch_or(EB_MEMORY_ERROR, Ordering::Relaxed);

        memory_error_after_fork_child();
        assert_eq!(
            load() & EB_MEMORY_ERROR,
            0,
            "the thread that called fork owes nothing, so the child owes nothing"
        );
        assert_eq!(MEMORY_ERROR_OWERS.load(Ordering::Acquire), 0);

        // The survivor's own debt is the half that is kept.
        set_memory_error();
        MEMORY_ERROR_OWERS.fetch_add(1, Ordering::AcqRel);
        memory_error_after_fork_child();
        assert_ne!(
            load() & EB_MEMORY_ERROR,
            0,
            "an exception owed to the surviving thread is still owed after the fork"
        );
        assert!(
            take_memory_error(),
            "and it is still that thread's to raise"
        );
        assert_eq!(load() & EB_MEMORY_ERROR, 0);
    }

    fn int(value: i64) -> crate::operand::Operand {
        crate::operand::Operand::const_(Const::Int(value))
    }

    /// The compiled poll compares the whole word against `JIT_BREAKER_FLOOR`
    /// where it used to mask and then test the remainder. That is only the
    /// same question while the non-breaker bits stay below the floor, so
    /// assert the equivalence over every word the flags can build rather than
    /// over the handful the other tests happen to produce.
    #[test]
    fn the_floor_compare_answers_the_mask_test_for_every_representable_word() {
        for word in 0..=ALL_EVAL_BREAKER_BITS {
            assert_eq!(
                word >= JIT_BREAKER_FLOOR,
                word & JIT_BREAKER_MASK != 0,
                "word {word:#b} disagrees"
            );
        }
    }

    /// Build the recorded back-edge poll over `load_addr`, which the caller
    /// varies to separate "this word" from "some other raw load".
    fn poll_guard(load_addr: usize) -> Op {
        let load = bind(Op::new(OpCode::RawLoadI, &[int(load_addr as i64), int(0)]));
        let armed = bind(Op::new(
            OpCode::UintGe,
            &[load, int(JIT_BREAKER_FLOOR as i64)],
        ));
        Op::new(OpCode::GuardFalse, &[armed])
    }

    /// The counter split turns on this predicate alone, so it has to separate
    /// the poll from every other guard the same trace records. Only the
    /// address published for this word matches: a raw load of a neighbouring
    /// address reaches the same opcode chain, and a data guard reaches none of
    /// it.
    ///
    /// `publish_addr` is idempotent and writes only the address holder, not the
    /// word, so this leaves the process-global flag state alone.
    #[test]
    fn only_a_poll_of_this_words_address_is_recognised() {
        publish_addr();
        let addr = eval_breaker_word_addr();
        assert_ne!(addr, 0, "publish_addr must make the address readable");

        assert!(is_back_edge_poll_guard(&poll_guard(addr)));
        // Same shape, a different word: the JIT records other raw loads.
        assert!(!is_back_edge_poll_guard(&poll_guard(
            addr + EVAL_BREAKER_WORD_SIZE
        )));
        // A guard on traced values — the population whose failures the
        // `guard_failures` total is meant to describe.
        let value = bind(Op::new(OpCode::IntAdd, &[int(1), int(2)]));
        let is_true = bind(Op::new(OpCode::IntIsTrue, &[value]));
        assert!(!is_back_edge_poll_guard(&Op::new(
            OpCode::GuardTrue,
            &[is_true]
        )));
        // A control-flow guard carries no condition operand at all.
        assert!(!is_back_edge_poll_guard(&Op::new(
            OpCode::GuardNotForced,
            &[]
        )));
    }

    /// `ActionFlag.decrement_ticker` stores an `isize`; the compiled back-edge
    /// load/store descriptor must use the same width.
    #[test]
    fn action_ticker_word_is_pointer_width() {
        assert_eq!(ACTION_TICKER_WORD_SIZE, size_of::<isize>());
    }

    /// One request arms the poll once: the taker reports it, clears it, and the
    /// next taker sees nothing. A taker that failed to clear would keep failing
    /// the back-edge guard and deopt the loop on every iteration. Taking it
    /// must also leave the other two bits alone — they address disjoint parts
    /// of the same word.
    ///
    /// The word is a process-global, so this is deliberately the only test in
    /// the crate that touches it.
    #[test]
    fn gc_request_is_taken_exactly_once_and_leaves_the_other_bits() {
        assert!(!take_gc(), "no request armed at the start of the test");
        set_async();
        set_stw();
        set_gc();
        assert_ne!(load() & EB_GC, 0);
        assert!(take_gc());
        assert_eq!(load() & EB_GC, 0);
        assert!(!take_gc());
        assert_eq!(load() & (EB_ASYNC | EB_STW), EB_ASYNC | EB_STW);
        clear_async();
        clear_stw();
    }
}
