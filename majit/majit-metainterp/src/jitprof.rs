//! Port of `rpython/jit/metainterp/jitprof.py Profiler`.
//!
//! RPython carries `counters: list[int]` indexed by `Counters.*` (rlib/jit.py:
//! 1414-1442) and a separate `calls: int` for the CALL+RECORDED_OPS path.
//! Pyre stores each counter as its own `AtomicUsize` field on
//! [`JitProfiler`] so cross-crate callers (TraceCtx in `pyre-jit-trace`,
//! heapcache in `majit-trace`, the vector pass in `optimizeopt`) can hit
//! `MetaInterpStaticData.profiler` through the shared `Arc` without any
//! extra synchronisation.
//!
//! `Ordering::Relaxed` is sufficient for every counter/timer total: there is
//! no causal relationship between any two updates, and we only ever publish
//! totals via [`JitProfiler::snapshot`] which itself is `Relaxed`.
//!
//! The `t1` / `current` event stack mirrors PyPy's `self.t1`/`self.current`
//! instance fields (`jitprof.py`) — held behind a per-`JitProfiler`
//! `Mutex<TimingState>` so concurrent threads sharing the profiler via
//! `Arc` serialize on the same lock the GIL gives PyPy.

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub use wasm_clock::Instant;

/// Monotonic substitute for `std::time::Instant` on wasm32-unknown-unknown,
/// which has no clock (`Instant::now()` there panics). The profiler only needs
/// non-decreasing values to charge relative time against, so each `now()`
/// advances a global counter; the resulting `MAJIT_STATS` timings are arbitrary
/// but internally consistent and never panic. Mirrors the
/// `Instant::now()` / `saturating_duration_since` surface the timer uses, so
/// the timing code below is platform-agnostic.
#[cfg(target_arch = "wasm32")]
pub mod wasm_clock {
    use core::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static TICK_NS: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy, Debug)]
    pub struct Instant(u64);

    impl Instant {
        pub fn now() -> Self {
            Instant(TICK_NS.fetch_add(1, Ordering::Relaxed))
        }

        pub fn saturating_duration_since(self, earlier: Instant) -> Duration {
            Duration::from_nanos(self.0.saturating_sub(earlier.0))
        }
    }
}

use majit_backend::CpuTotalTracker;
use majit_ir::OpCode;

use crate::pyjitpl::counters;

/// jitprof.py `BaseProfiler`.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseProfiler;

/// jitprof.py `EmptyProfiler`.
///
/// Used by upstream tests and by `WarmRunnerDesc(...,
/// ProfilerClass=EmptyProfiler)` when statistics collection is disabled.
/// Pyre's runtime path currently installs [`Profiler`] unconditionally, but
/// this no-op surface keeps the translated module shape aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyProfiler {
    /// jitprof.py `initialized = True`.
    pub initialized: bool,
}

impl Default for EmptyProfiler {
    fn default() -> Self {
        Self { initialized: true }
    }
}

impl EmptyProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&self) {}

    pub fn finish(&self) {}

    pub fn start_tracing(&self) {}

    pub fn end_tracing(&self) {}

    pub fn start_backend(&self) {}

    pub fn end_backend(&self) {}

    pub fn count(&self, _kind: i32, _inc: usize) {}

    pub fn count_ops(&self, _opnum: OpCode, _kind: i32) {}

    pub fn get_counter(&self, _num: i32) -> usize {
        0
    }

    pub fn get_times(&self, _num: i32) -> f64 {
        0.0
    }
}

/// `self.t1` + `self.current` from `jitprof.py:56,60`.  Kept behind a
/// `Mutex` on the owning [`JitProfiler`] so the GIL-protected
/// list/scalar pair in upstream becomes a critical section here.
///
/// **Single-thread contract.**  PyPy's GIL serialises every
/// `_start`/`_end` call, so `current` behaves as a strict LIFO stack
/// with no cross-thread interleaving.  Pyre cannot rely on GIL —
/// every `JitProfiler` is shared via `Arc` and JIT operations may
/// touch the same profiler from multiple worker threads.  The
/// `Mutex` only protects each individual `_start`/`_end` push/pop;
/// between calls the lock is released, so two threads opening
/// overlapping scopes would interleave the stack and trip
/// `BROKEN PROFILER DATA!` on the second drop.  Callers MUST
/// serialise profiler `start_*`/`end_*` at a higher level (typically
/// by holding a JIT-wide lock); the [`owner_thread`] field below
/// catches accidental sharing in debug builds.
///
/// [`owner_thread`]: TimingState::owner_thread
#[derive(Default, Debug)]
struct TimingState {
    /// `self.starttime` (`jitprof.py`) — fixed profile start mark;
    /// `_print_stats`'s TOTAL line is `tk - starttime`.
    starttime: Option<Instant>,
    /// `self.tk` (`jitprof.py`) — set by `finish()` just before
    /// `print_stats()`.
    tk: Option<Instant>,
    /// `self.t1` (`jitprof.py:56`) — timer baseline for the next
    /// `_start`/`_end` to charge time against.
    t1: Option<Instant>,
    /// `self.current` (`jitprof.py:60,69`) — nested event stack.  Each
    /// entry is a `Counters.*` id matching the matching `_start(event)`
    /// push at `jitprof.py`.
    current: Vec<i32>,
    /// Set to the thread that first pushes onto an empty stack; cleared
    /// once the stack drains back to empty.  Same-thread re-entry is
    /// fine (the GIL-equivalent invariant only requires serialisation,
    /// not single-thread exclusivity across the profiler lifetime); a
    /// `_start`/`_end` from a *different* thread while the stack is
    /// non-empty trips a debug-build panic.  Detects the cross-thread
    /// interleaving bug at the point it would corrupt state instead of
    /// at the later `BROKEN PROFILER DATA!` mismatch.
    owner_thread: Option<std::thread::ThreadId>,
}

/// jitprof.py `Profiler` — every `Counters.*` slot is one
/// `AtomicUsize`, plus the standalone `calls` counter that
/// `count_ops` increments on the CALL_*+RECORDED_OPS path
/// (jitprof.py).
///
/// `field_for_kind` maps a `Counters.*` id (see `pyjitpl::counters`) to
/// the matching `AtomicUsize`; unknown ids are silently ignored,
/// matching upstream's permissive `self.counters[kind] += 1` (an
/// out-of-range id raises `IndexError` upstream, but that has only ever
/// fired for hand-rolled counter ids that are not part of the
/// canonical `Counters` enum — pyre never produces such ids).
/// The reason-keyed `Counters.ABORT_*` tallies in a fixed order, with the
/// short label each one carries in a `[jit-stats]` line.
///
/// `loops_aborted` is one number, and the split that explains it normally
/// comes from `MAJIT_LOG`'s summary — a guest-side knob. The wasm32 guest has
/// no environment, so on that backend the knob is inert and a `loops_aborted`
/// regression arrives with no reason attached. Enumerating the tallies here
/// lets the host read them by index and print the same breakdown.
///
/// Order is the contract with that host: append only, never reorder.
///
/// `ABORT_BRIDGE` is labelled for what it actually counts here, not for its
/// name. `AbortReason::Generic` maps to the same integer 13, and that is the
/// value `jitdriver`'s reason ladder falls back to whenever a `TraceAction::
/// Abort` arrives with no `SwitchToBlackhole`, no staged reason and a trace
/// that is not too long — which is every walker decline, without exception.
/// So a nonzero tally here is overwhelmingly "unclassified", and reading it as
/// "a bridge aborted" sends the next reader looking for bridge activity that is
/// not there.
///
/// `ABORT_FORCE_QUASIIMMUT` has two producers, both sitting on the residual-call
/// boundary, because that is where a quasi-immutable write lands here:
/// `try_walker_force_quasi_immut_namespace_write` and
/// `try_walker_force_quasi_immut_mapdict_write`
/// (`pyre-jit-trace/src/jitcode_dispatch/residual_call.rs`). Each calls
/// `note_force_quasi_immut_abort`, which stages the reason so the single
/// `aborted_tracing` that follows tallies it here instead of in the `Generic`
/// catch-all — the stand-in for the reason
/// `opimpl_jit_force_quasi_immutable` (`pyjitpl.py`) carries on the
/// `SwitchToBlackhole` instance it raises.
///
/// The mapdict producer mirrors `opimpl_jit_force_quasi_immutable` and runs
/// whenever a traced mapdict write changes a quasi-immutable attribute.  The
/// namespace producer remains independently live; for example,
/// `MAJIT_STATS=1 PYRE_MC_DIAG=1` on
/// `bench/synth/trace_too_long_effect_replay.py` reaches its
/// `abrt_force_qmut` tally.
pub const ABORT_COUNTER_KINDS: &[(i32, &str)] = &[
    (counters::ABORT_TOO_LONG, "too_long"),
    (counters::ABORT_BRIDGE, "bridge_or_generic"),
    (counters::ABORT_BAD_LOOP, "bad_loop"),
    (counters::ABORT_ESCAPE, "escape"),
    (counters::ABORT_FORCE_QUASIIMMUT, "force_quasiimmut"),
    (counters::ABORT_SEGMENTED_TRACE, "segmented_trace"),
];

#[derive(Default, Debug)]
pub struct JitProfiler {
    /// jit.py `Counters.TRACING` — RPython tracks this as wall-clock
    /// time + entry count via `_start`/`_end` (jitprof.py).
    pub tracing: AtomicUsize,
    /// jit.py `Counters.BACKEND` — same shape as TRACING.
    pub backend: AtomicUsize,
    /// Accumulated nanoseconds for `Counters.TRACING`.
    pub tracing_time_ns: AtomicU64,
    /// Accumulated nanoseconds for `Counters.BACKEND`.
    pub backend_time_ns: AtomicU64,
    /// jit.py `Counters.OPS` — every executed op
    /// (`execute_and_record_varargs` / `execute_and_record`,
    /// pyjitpl.py/2645).
    pub ops: AtomicUsize,
    /// jit.py `Counters.HEAPCACHED_OPS` — folded-away ops that the
    /// heapcache resolved without recording (pyjitpl.py:388/397/562/...).
    pub heapcached_ops: AtomicUsize,
    /// jit.py `Counters.RECORDED_OPS` — ops that survived the
    /// heapcache and reached `_record_helper` / `_record_helper_varargs`
    /// (pyjitpl.py/2669).
    pub recorded_ops: AtomicUsize,
    /// jit.py `Counters.GUARDS` — guards counted by the trace
    /// recorder (pyjitpl.py:2581).
    pub guards: AtomicUsize,
    /// jit.py `Counters.OPT_OPS` — every op the optimizer emits
    /// (optimizer.py inside `_emit_operation`).
    pub opt_ops: AtomicUsize,
    /// jit.py `Counters.OPT_GUARDS` — guards emitted by the
    /// optimizer (optimizer.py:629).
    pub opt_guards: AtomicUsize,
    /// jit.py `Counters.OPT_GUARDS_SHARED` — guards that share
    /// resume data with a previous guard via descriptor fusion
    /// (optimizer.py:673-674).
    pub opt_guards_shared: AtomicUsize,
    /// jit.py `Counters.OPT_FORCINGS`.
    pub opt_forcings: AtomicUsize,
    /// jit.py `Counters.OPT_VECTORIZE_TRY` — entries into the
    /// vector pass (vector.py:139).
    pub opt_vectorize_try: AtomicUsize,
    /// jit.py `Counters.OPT_VECTORIZED` — successful vectorise
    /// (vector.py:146).
    pub opt_vectorized: AtomicUsize,
    /// jit.py `Counters.ABORT_TOO_LONG`.
    pub abort_too_long: AtomicUsize,
    /// jit.py `Counters.ABORT_BRIDGE`.
    pub abort_bridge: AtomicUsize,
    /// jit.py `Counters.ABORT_BAD_LOOP`.
    pub abort_bad_loop: AtomicUsize,
    /// jit.py `Counters.ABORT_ESCAPE`.
    pub abort_escape: AtomicUsize,
    /// jit.py `Counters.ABORT_FORCE_QUASIIMMUT`.
    pub abort_force_quasiimmut: AtomicUsize,
    /// jit.py `Counters.ABORT_SEGMENTED_TRACE`.
    pub abort_segmented_trace: AtomicUsize,
    /// jit.py `Counters.FORCE_VIRTUALIZABLES`.
    pub force_virtualizables: AtomicUsize,
    /// jit.py `Counters.NVIRTUALS`.
    pub nvirtuals: AtomicUsize,
    /// jit.py `Counters.NVHOLES`.
    pub nvholes: AtomicUsize,
    /// jit.py `Counters.NVREUSED`.
    pub nvreused: AtomicUsize,
    /// jitprof.Profiler.calls — `count_ops` increments this when the op
    /// is a CALL_* and `kind == RECORDED_OPS` (jitprof.py).
    pub calls: AtomicUsize,
    /// pyjitpl.py `_setup_once` guard — `if not
    /// self.profiler.initialized: self.profiler.start(); ...
    /// initialized = True`.  RPython keeps this flag separate from
    /// `Profiler.start()`: `start()` always resets counters, while
    /// `_setup_once` decides whether to call it.
    pub initialized: AtomicBool,
    /// `jitprof.py:56,60 self.t1 / self.current` — instance-owned
    /// timing state.  Held behind `Mutex` so threads sharing the
    /// profiler via `Arc` cannot race on `_start`/`_end`; PyPy's GIL
    /// gives the same exclusion.
    timing: Mutex<TimingState>,
    /// `self.cpu.tracker` (`jitprof.py:105-106`) — PyPy reads
    /// `Counters.TOTAL_COMPILED_*` / `TOTAL_FREED_*` via
    /// `self.cpu.tracker.total_*`.  Pyre's `JitProfiler` holds an
    /// `Arc<CpuTotalTracker>` so reads through
    /// [`get_counter`](Self::get_counter) and writes through
    /// [`inc_freed_loop`](Self::inc_freed_loop) /
    /// [`add_freed_bridges`](Self::add_freed_bridges) hit the same
    /// store the paired [`majit_backend::Backend`] writes to via
    /// [`record_compiled_loop_token`](majit_backend::record_compiled_loop_token).
    /// `MetaInterp::new` rebinds this field to share the backend's
    /// `Arc` once both are constructed (see
    /// [`set_cpu_tracker`](Self::set_cpu_tracker)) so the metainterp
    /// pair behaves like PyPy's per-CPU tracker.
    cpu_tracker: Mutex<Arc<CpuTotalTracker>>,
}

/// Upstream public name for [`JitProfiler`].
pub type Profiler = JitProfiler;

impl JitProfiler {
    /// jitprof.py `Profiler.start`.
    ///
    /// Not idempotent by design: upstream `Profiler.start()` resets
    /// `self.counters` and `self.calls` every time it is called.  The
    /// one-shot guard lives at the caller (`pyjitpl.py:2300-2302`
    /// `_setup_once`: `if not self.profiler.initialized: ...`), not
    /// inside `start()`.
    pub fn start(&self) {
        for field in [
            &self.tracing,
            &self.backend,
            &self.ops,
            &self.heapcached_ops,
            &self.recorded_ops,
            &self.guards,
            &self.opt_ops,
            &self.opt_guards,
            &self.opt_guards_shared,
            &self.opt_forcings,
            &self.opt_vectorize_try,
            &self.opt_vectorized,
            &self.abort_too_long,
            &self.abort_bridge,
            &self.abort_bad_loop,
            &self.abort_escape,
            &self.abort_force_quasiimmut,
            &self.abort_segmented_trace,
            &self.force_virtualizables,
            &self.nvirtuals,
            &self.nvholes,
            &self.nvreused,
            &self.calls,
            // `cpu.tracker` counters (`TOTAL_COMPILED_*` /
            // `TOTAL_FREED_*`) live on the per-instance `cpu_tracker`
            // Arc bound to the backend's `CpuTotalTracker` — they
            // survive `Profiler.start()` (which only resets
            // `self.counters` and `self.calls`, jitprof.py).
        ] {
            field.store(0, Ordering::Relaxed);
        }
        self.tracing_time_ns.store(0, Ordering::Relaxed);
        self.backend_time_ns.store(0, Ordering::Relaxed);
        // `jitprof.py start()`:
        //   self.starttime = self.timer()
        //   self.t1 = self.starttime
        //   ...
        //   self.current = []
        let now = Instant::now();
        let mut state = self.timing.lock();
        state.starttime = Some(now);
        state.tk = None;
        state.t1 = Some(now);
        state.current.clear();
        state.owner_thread = None;
    }

    /// jitprof.py `Profiler.finish`.
    ///
    /// ```text
    ///  def finish(self):
    ///      self.tk = self.timer()
    ///      self.print_stats()
    /// ```
    pub fn finish(&self) {
        {
            let mut state = self.timing.lock();
            state.tk = Some(Instant::now());
        }
        self.print_stats();
    }

    /// jitprof.py `Profiler.print_stats` — emit the
    /// `jit-summary` section (visible under `MAJIT_LOG`).
    pub fn print_stats(&self) {
        crate::debug::debug_start("jit-summary");
        if crate::debug::have_debug_prints() {
            self._print_stats();
        }
        crate::debug::debug_stop("jit-summary");
    }

    /// jitprof.py `Profiler._print_stats` — line set, order,
    /// and layout are upstream's.
    fn _print_stats(&self) {
        let snap = self.snapshot();
        let (starttime, tk) = {
            let state = self.timing.lock();
            (state.starttime, state.tk)
        };
        let ns_to_secs = |ns: u64| ns as f64 / 1e9;
        crate::debug::debug_print(&fmt_line_time(
            "Tracing",
            snap.tracing,
            ns_to_secs(self.tracing_time_ns.load(Ordering::Relaxed)),
        ));
        crate::debug::debug_print(&fmt_line_time(
            "Backend",
            snap.backend,
            ns_to_secs(self.backend_time_ns.load(Ordering::Relaxed)),
        ));
        let total = match (starttime, tk) {
            (Some(start), Some(end)) => end.saturating_duration_since(start).as_secs_f64(),
            _ => 0.0,
        };
        crate::debug::debug_print(&format!("TOTAL:      \t\t{total:.6}"));
        let mut intline = |label: &str, value: usize| {
            crate::debug::debug_print(&fmt_intline(label, value));
        };
        intline("ops", snap.ops);
        intline("heapcached ops", snap.heapcached_ops);
        intline("recorded ops", snap.recorded_ops);
        intline("  calls", snap.calls);
        intline("guards", snap.guards);
        intline("opt ops", snap.opt_ops);
        intline("opt guards", snap.opt_guards);
        intline("opt guards shared", snap.opt_guards_shared);
        intline("forcings", snap.opt_forcings);
        intline("abort: trace too long", snap.abort_too_long);
        intline("abort: compiling", snap.abort_bridge);
        intline("abort: vable escape", snap.abort_escape);
        intline("abort: bad loop", snap.abort_bad_loop);
        intline("abort: force quasi-immut", snap.abort_force_quasiimmut);
        intline("abort: segmenting trace", snap.abort_segmented_trace);
        intline("virtualizables forced", snap.force_virtualizables);
        intline("nvirtuals", snap.nvirtuals);
        intline("nvholes", snap.nvholes);
        intline("nvreused", snap.nvreused);
        intline("vecopt tried", snap.opt_vectorize_try);
        intline("vecopt success", snap.opt_vectorized);
        // jitprof.py `if cpu is not None:` — pyre's tracker Arc
        // is always bound (a fresh one before `set_cpu_tracker`).
        let tracker = self.cpu_tracker.lock().clone();
        intline(
            "Total # of loops",
            tracker.total_compiled_loops.load(Ordering::Relaxed),
        );
        intline(
            "Total # of bridges",
            tracker.total_compiled_bridges.load(Ordering::Relaxed),
        );
        intline(
            "Freed # of loops",
            tracker.total_freed_loops.load(Ordering::Relaxed),
        );
        intline(
            "Freed # of bridges",
            tracker.total_freed_bridges.load(Ordering::Relaxed),
        );
    }

    /// jitprof.py `Profiler.start_tracing`.
    pub fn start_tracing(&self) {
        self.start_event(counters::TRACING);
    }

    /// jitprof.py `Profiler.end_tracing`.
    pub fn end_tracing(&self) {
        self.end_event(counters::TRACING);
    }

    /// jitprof.py `Profiler.start_backend`.
    pub fn start_backend(&self) {
        self.start_event(counters::BACKEND);
    }

    /// jitprof.py `Profiler.end_backend`.
    pub fn end_backend(&self) {
        self.end_event(counters::BACKEND);
    }

    /// jitprof.py `Profiler.count_ops(opnum, kind=Counters.OPS)`.
    ///
    /// ```python
    /// def count_ops(self, opnum, kind=Counters.OPS):
    ///     self.counters[kind] += 1
    ///     if OpHelpers.is_call(opnum) and kind == Counters.RECORDED_OPS:
    ///         self.calls += 1
    /// ```
    ///
    /// `kind` is a [`crate::pyjitpl::counters`] id. Unknown ids are a
    /// silent no-op (see field doc).  `TOTAL_COMPILED_*` /
    /// `TOTAL_FREED_*` panic in both debug and release: PyPy
    /// `Profiler.counters` is sized `Counters.ncounters = 22` so
    /// `self.counters[TOTAL_*]` raises `IndexError`.  Pyre uses
    /// `assert!` so the release path crashes too — the unknown-id
    /// silent no-op below is for truly out-of-range ids, while
    /// `TOTAL_*` has a well-defined alternate sink (the backend's
    /// [`CpuTotalTracker`]); silently routing a stray write to
    /// nowhere would mask the bug.
    pub fn count_ops(&self, opnum: OpCode, kind: i32) {
        assert!(
            !is_cpu_tracker_kind(kind),
            "Profiler.count_ops({kind}) called with a CPU-total id; \
             route through CpuTotalTracker directly (PyPy raises \
             IndexError here)",
        );
        if let Some(field) = self.field_for_kind(kind) {
            field.fetch_add(1, Ordering::Relaxed);
        }
        if opnum.is_call() && kind == counters::RECORDED_OPS {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// jitprof.py `Profiler.count(kind, inc=1)`.
    ///
    /// ```python
    /// def count(self, kind, inc=1):
    ///     self.counters[kind] += inc
    /// ```
    ///
    /// Used for non-op events (ABORT_*, NV*, OPT_VECTORIZE_*, ...).
    /// Unknown ids are a silent no-op (matching the OPS variant above).
    /// `TOTAL_COMPILED_*` / `TOTAL_FREED_*` panic in both debug and
    /// release: PyPy `self.counters[TOTAL_*]` is out-of-range
    /// (`Counters.ncounters = 22`) and raises `IndexError`, so pyre
    /// uses `assert!` to crash on both build profiles.  Use the
    /// backend's [`CpuTotalTracker`] directly (or
    /// [`Self::inc_freed_loop`] / [`Self::add_freed_bridges`]) when a
    /// tracker bump is actually intended.
    pub fn count(&self, kind: i32, inc: usize) {
        assert!(
            !is_cpu_tracker_kind(kind),
            "Profiler.count({kind}) called with a CPU-total id; \
             route through CpuTotalTracker directly (PyPy raises \
             IndexError here)",
        );
        if let Some(field) = self.field_for_kind(kind) {
            field.fetch_add(inc, Ordering::Relaxed);
        }
    }

    /// jitprof.py `Profiler.get_counter(num)` — single-counter
    /// readback via `Counters.*` id.  PyPy routes `TOTAL_COMPILED_*` /
    /// `TOTAL_FREED_*` (ids 22..25) to `self.cpu.tracker.total_*`; pyre
    /// reads from `self.cpu_tracker` for the same four ids and from
    /// `self` for everything else.  Unknown ids return `None`.
    pub fn get_counter(&self, kind: i32) -> Option<usize> {
        if is_cpu_tracker_kind(kind) {
            return Some(
                self.with_cpu_tracker(|t| cpu_tracker_field(t, kind).load(Ordering::Relaxed)),
            );
        }
        self.field_for_kind(kind)
            .map(|field| field.load(Ordering::Relaxed))
    }

    /// Read the `i`th [`ABORT_COUNTER_KINDS`] tally.  Out of range reads 0,
    /// so a reader holding a longer list than the profiler it queries — the
    /// wasm host, which knows the labels but reaches the counters only by
    /// index across the module boundary — degrades instead of trapping.
    pub fn abort_diag(&self, i: usize) -> u64 {
        ABORT_COUNTER_KINDS
            .get(i)
            .and_then(|&(kind, _)| self.get_counter(kind))
            .unwrap_or(0) as u64
    }

    /// `cpu.tracker.total_freed_loops += 1` parity.  Production bumps
    /// live in `CompiledLoopToken::drop`; this method remains for tests.
    /// Hits `self.cpu_tracker` so the paired backend (rebound via
    /// [`Self::set_cpu_tracker`]) and profiler share the same per-CPU
    /// instance.
    pub fn inc_freed_loop(&self) {
        self.with_cpu_tracker(|t| t.total_freed_loops.fetch_add(1, Ordering::Relaxed));
    }

    /// `cpu.tracker.total_freed_bridges += n` parity.  Fired from the
    /// memory manager when an evicted token carries `n` bridges; PyPy's
    /// `cpu.free_loop_and_bridges` bumps the tracker once per bridge.
    pub fn add_freed_bridges(&self, n: usize) {
        self.with_cpu_tracker(|t| t.total_freed_bridges.fetch_add(n, Ordering::Relaxed));
    }

    /// Rebind `self.cpu_tracker` to the backend's
    /// [`CpuTotalTracker`] so the profiler and backend share one
    /// counter sink.  `MetaInterp::new` calls this once the backend is
    /// available, mirroring PyPy where `Profiler` reads through
    /// `self.cpu.tracker` (jitprof.py) — `self.cpu` being the
    /// backend's `AbstractCPU` instance, not a process global.
    pub fn set_cpu_tracker(&self, tracker: Arc<CpuTotalTracker>) {
        let mut slot = self.cpu_tracker.lock();
        *slot = tracker;
    }

    /// Clone the current `Arc<CpuTotalTracker>` (cheap refcount bump)
    /// and run `f` against it outside the mutex so the lock is held
    /// only for the swap window.  Callers (`inc_freed_loop`,
    /// `add_freed_bridges`, `get_counter`) avoid serialising on the
    /// mutex during the atomic op itself.
    fn with_cpu_tracker<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&CpuTotalTracker) -> R,
    {
        let tracker = {
            let slot = self.cpu_tracker.lock();
            Arc::clone(&slot)
        };
        f(&tracker)
    }

    /// jitprof.py `Profiler.get_times(num)` — seconds.
    pub fn get_times(&self, kind: i32) -> Option<f64> {
        self.time_field_for_kind(kind)
            .map(|field| field.load(Ordering::Relaxed) as f64 / 1_000_000_000.0)
    }

    /// Snapshot every counter at a moment.
    ///
    /// Each load is `Relaxed`, so the returned snapshot is **not**
    /// guaranteed to be a single coherent read across counters — only
    /// each individual counter's value is consistent with itself.
    /// Mirrors RPython where `_print_stats` (jitprof.py) reads
    /// each `cnt[Counters.X]` one-by-one without any locking.
    pub fn snapshot(&self) -> JitProfilerSnapshot {
        JitProfilerSnapshot {
            tracing: self.tracing.load(Ordering::Relaxed),
            backend: self.backend.load(Ordering::Relaxed),
            ops: self.ops.load(Ordering::Relaxed),
            heapcached_ops: self.heapcached_ops.load(Ordering::Relaxed),
            recorded_ops: self.recorded_ops.load(Ordering::Relaxed),
            guards: self.guards.load(Ordering::Relaxed),
            opt_ops: self.opt_ops.load(Ordering::Relaxed),
            opt_guards: self.opt_guards.load(Ordering::Relaxed),
            opt_guards_shared: self.opt_guards_shared.load(Ordering::Relaxed),
            opt_forcings: self.opt_forcings.load(Ordering::Relaxed),
            opt_vectorize_try: self.opt_vectorize_try.load(Ordering::Relaxed),
            opt_vectorized: self.opt_vectorized.load(Ordering::Relaxed),
            abort_too_long: self.abort_too_long.load(Ordering::Relaxed),
            abort_bridge: self.abort_bridge.load(Ordering::Relaxed),
            abort_bad_loop: self.abort_bad_loop.load(Ordering::Relaxed),
            abort_escape: self.abort_escape.load(Ordering::Relaxed),
            abort_force_quasiimmut: self.abort_force_quasiimmut.load(Ordering::Relaxed),
            abort_segmented_trace: self.abort_segmented_trace.load(Ordering::Relaxed),
            force_virtualizables: self.force_virtualizables.load(Ordering::Relaxed),
            nvirtuals: self.nvirtuals.load(Ordering::Relaxed),
            nvholes: self.nvholes.load(Ordering::Relaxed),
            nvreused: self.nvreused.load(Ordering::Relaxed),
            calls: self.calls.load(Ordering::Relaxed),
            tracing_time_ns: self.tracing_time_ns.load(Ordering::Relaxed),
            backend_time_ns: self.backend_time_ns.load(Ordering::Relaxed),
        }
    }

    /// Panic-safe RAII pairing matching `pyjitpl.py compile_and_run_once / 2914-2935`:
    ///
    /// ```python
    /// debug_start('jit-tracing')      # outer
    /// profiler.start_tracing()        # inner
    /// try:
    ///     ...
    /// finally:
    ///     profiler.end_tracing()      # inner close
    ///     debug_stop('jit-tracing')   # outer close
    /// ```
    ///
    /// Construction fires `debug_start` then `start_tracing`; drop
    /// fires `end_tracing` then `debug_stop`.  Panics inside the body
    /// still unwind through both layers, so the `current` stack and
    /// the debug section stay balanced.
    pub fn enter_tracing(&self) -> ProfilerEventGuard<'_> {
        let channel = debug_channel_for_event(counters::TRACING);
        if let Some(ch) = channel {
            crate::debug::debug_start(ch);
        }
        // Rollback guard: the debug section is open; `start_tracing`
        // takes the timing-state mutex, which can panic on poison.
        // Without the guard, that panic would leave the section open
        // for later `debug_stop` mismatch panics.  Disarm via
        // `mem::forget` once both opens succeed so the returned
        // `ProfilerEventGuard::drop` owns the normal-path close.
        let rollback = TracingOpenRollback { channel };
        self.start_tracing();
        std::mem::forget(rollback);
        ProfilerEventGuard {
            profiler: self,
            event: counters::TRACING,
            nesting: GuardNesting::DebugOuter,
        }
    }

    /// Panic-safe RAII pairing matching `compile.py:532-546 / 589-599`:
    ///
    /// ```python
    /// metainterp_sd.profiler.start_backend()   # outer
    /// debug_start('jit-backend')               # inner
    /// try:
    ///     ...
    /// finally:
    ///     debug_stop('jit-backend')            # inner close
    /// metainterp_sd.profiler.end_backend()     # outer close
    /// ```
    ///
    /// Note that backend nesting is **reversed** relative to tracing:
    /// `profiler.start_backend()` opens the outer scope here, while
    /// `debug_start('jit-tracing')` opens the outer scope in
    /// [`Self::enter_tracing`].  PyPy uses both orders depending on the
    /// callsite — this guard matches each one exactly.
    pub fn enter_backend(&self) -> ProfilerEventGuard<'_> {
        self.start_backend();
        // Rollback guard: the backend event is now open, but
        // `debug_start` may still panic on a nesting violation
        // (`majit-ir`'s `debug.rs` `debug_stop` strict-stop guard).  Without the guard, a
        // panic between the two opens would leave the profiler stack
        // dirty for later calls.  The local guard fires
        // `end_backend()` on unwind; `mem::forget` disarms it once
        // both opens succeed so the returned RAII guard owns the
        // normal-path close.
        let rollback = BackendOpenRollback { profiler: self };
        if let Some(channel) = debug_channel_for_event(counters::BACKEND) {
            crate::debug::debug_start(channel);
        }
        std::mem::forget(rollback);
        ProfilerEventGuard {
            profiler: self,
            event: counters::BACKEND,
            nesting: GuardNesting::ProfilerOuter,
        }
    }

    fn start_event(&self, event: i32) {
        // `jitprof.py _start(event)` — profiler bookkeeping only.
        // The matching `debug_start(channel)` lives at the caller
        // (PyPy convention; mirrored by [`enter_tracing`] /
        // [`enter_backend`]).
        //   t0 = self.t1
        //   self.t1 = self.timer()
        //   if self.current:
        //       self.times[self.current[-1]] += self.t1 - t0
        //   self.counters[event] += 1
        //   self.current.append(event)
        let now = Instant::now();
        let mut state = self.timing.lock();
        check_or_claim_owner_thread(&mut state, "start_event");
        if let (Some(t1), Some(&top_event)) = (state.t1, state.current.last()) {
            self.add_time(top_event, now.saturating_duration_since(t1));
        }
        state.t1 = Some(now);
        self.count(event, 1);
        state.current.push(event);
    }

    fn end_event(&self, event: i32) {
        // `jitprof.py _end(event)` — pop-first, then validate.
        // The matching `debug_stop(channel)` lives at the caller
        // (PyPy convention; mirrored by [`ProfilerEventGuard::drop`]).
        //   t0 = self.t1
        //   self.t1 = self.timer()
        //   if not self.current:
        //       debug_print("BROKEN PROFILER DATA!"); return
        //   ev1 = self.current.pop()
        //   if ev1 != event:
        //       debug_print("BROKEN PROFILER DATA!"); return
        //   self.times[ev1] += self.t1 - t0
        let now = Instant::now();
        let popped_event;
        let t0;
        {
            let mut state = self.timing.lock();
            check_or_claim_owner_thread(&mut state, "end_event");
            t0 = state.t1;
            state.t1 = Some(now);
            let Some(ev1) = state.current.pop() else {
                state.owner_thread = None;
                crate::debug::log_one("jit-profiler", "BROKEN PROFILER DATA!");
                return;
            };
            popped_event = ev1;
            // Drained back to empty — release ownership so the next
            // `start_event`/`end_event` from any thread can claim it.
            if state.current.is_empty() {
                state.owner_thread = None;
            }
        }
        if popped_event != event {
            crate::debug::log_one("jit-profiler", "BROKEN PROFILER DATA!");
            return;
        }
        if let Some(t1) = t0 {
            self.add_time(popped_event, now.saturating_duration_since(t1));
        }
    }

    fn add_time(&self, event: i32, elapsed: Duration) {
        let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        if let Some(field) = self.time_field_for_kind(event) {
            field.fetch_add(nanos, Ordering::Relaxed);
        }
    }

    fn field_for_kind(&self, kind: i32) -> Option<&AtomicUsize> {
        Some(match kind {
            counters::TRACING => &self.tracing,
            counters::BACKEND => &self.backend,
            counters::OPS => &self.ops,
            counters::HEAPCACHED_OPS => &self.heapcached_ops,
            counters::RECORDED_OPS => &self.recorded_ops,
            counters::GUARDS => &self.guards,
            counters::OPT_OPS => &self.opt_ops,
            counters::OPT_GUARDS => &self.opt_guards,
            counters::OPT_GUARDS_SHARED => &self.opt_guards_shared,
            counters::OPT_FORCINGS => &self.opt_forcings,
            counters::OPT_VECTORIZE_TRY => &self.opt_vectorize_try,
            counters::OPT_VECTORIZED => &self.opt_vectorized,
            counters::ABORT_TOO_LONG => &self.abort_too_long,
            counters::ABORT_BRIDGE => &self.abort_bridge,
            counters::ABORT_BAD_LOOP => &self.abort_bad_loop,
            counters::ABORT_ESCAPE => &self.abort_escape,
            counters::ABORT_FORCE_QUASIIMMUT => &self.abort_force_quasiimmut,
            counters::ABORT_SEGMENTED_TRACE => &self.abort_segmented_trace,
            counters::FORCE_VIRTUALIZABLES => &self.force_virtualizables,
            counters::NVIRTUALS => &self.nvirtuals,
            counters::NVHOLES => &self.nvholes,
            counters::NVREUSED => &self.nvreused,
            _ => return None,
        })
    }

    fn time_field_for_kind(&self, kind: i32) -> Option<&AtomicU64> {
        Some(match kind {
            counters::TRACING => &self.tracing_time_ns,
            counters::BACKEND => &self.backend_time_ns,
            _ => return None,
        })
    }
}

/// Rollback guard used inside [`JitProfiler::enter_backend`] for the
/// short window after `start_backend()` and before `debug_start` has
/// returned cleanly.  If `debug_start` panics, the unwind path fires
/// the matching `end_backend()` so the profiler stack does not stay
/// dirty.  Disarmed via `mem::forget` once both opens succeed.
struct BackendOpenRollback<'a> {
    profiler: &'a JitProfiler,
}

impl Drop for BackendOpenRollback<'_> {
    fn drop(&mut self) {
        self.profiler.end_backend();
    }
}

/// Rollback guard used inside [`JitProfiler::enter_tracing`] for the
/// short window after `debug_start("jit-tracing")` and before
/// `start_tracing()` has returned cleanly.  If `start_tracing` panics
/// (timing-state mutex poisoned), the unwind path fires the matching
/// `debug_stop` so the PYPYLOG category stack stays balanced.
struct TracingOpenRollback {
    channel: Option<&'static str>,
}

impl Drop for TracingOpenRollback {
    fn drop(&mut self) {
        if let Some(ch) = self.channel {
            crate::debug::debug_stop(ch);
        }
    }
}

/// Which scope is the outer one — tracing wraps `debug_start` around
/// the profiler call (`pyjitpl.py compile_and_run_once`), backend wraps the
/// profiler call around `debug_start` (`compile.py:532-546`).
/// [`ProfilerEventGuard::drop`] dispatches the LIFO close order based
/// on this flag so each callsite matches its PyPy counterpart exactly.
enum GuardNesting {
    /// `debug_start` is outer, `profiler.start_*` is inner.  Used by
    /// [`JitProfiler::enter_tracing`].
    DebugOuter,
    /// `profiler.start_*` is outer, `debug_start` is inner.  Used by
    /// [`JitProfiler::enter_backend`].
    ProfilerOuter,
}

/// RAII guard returned by [`JitProfiler::enter_tracing`] /
/// [`JitProfiler::enter_backend`].  Drops by firing both the
/// profiler-event close and the `debug_stop` close in the LIFO order
/// dictated by `GuardNesting`, so the profiler stack and the debug
/// section stay balanced even when the surrounding body panics.
#[must_use = "drop the guard to fire the paired end_* event"]
pub struct ProfilerEventGuard<'a> {
    profiler: &'a JitProfiler,
    event: i32,
    nesting: GuardNesting,
}

impl Drop for ProfilerEventGuard<'_> {
    fn drop(&mut self) {
        // LIFO close: inner scope first, then outer.  If `end_event`
        // detects a mismatch (broken-data path) we still emit
        // `debug_stop` because the matching `debug_start` was already
        // published; skipping it would leave the section open against
        // the strict nesting guard in [`crate::debug::debug_stop`].
        let channel = debug_channel_for_event(self.event);
        match self.nesting {
            // tracing: inner = profiler, outer = debug.
            GuardNesting::DebugOuter => {
                self.profiler.end_event(self.event);
                if let Some(ch) = channel {
                    crate::debug::debug_stop(ch);
                }
            }
            // backend: inner = debug, outer = profiler.
            GuardNesting::ProfilerOuter => {
                if let Some(ch) = channel {
                    crate::debug::debug_stop(ch);
                }
                self.profiler.end_event(self.event);
            }
        }
    }
}

/// `Counters.TOTAL_COMPILED_*` / `Counters.TOTAL_FREED_*` (jit.py)
/// — the four ids PyPy reads via `cpu.tracker` instead of
/// `self.counters`.  Used by [`JitProfiler::count`] /
/// [`JitProfiler::count_ops`] `debug_assert!` to flag callers that
/// would land an out-of-range write upstream.
fn is_cpu_tracker_kind(kind: i32) -> bool {
    matches!(
        kind,
        counters::TOTAL_COMPILED_LOOPS
            | counters::TOTAL_COMPILED_BRIDGES
            | counters::TOTAL_FREED_LOOPS
            | counters::TOTAL_FREED_BRIDGES
    )
}

/// Route `Counters.TOTAL_COMPILED_*` / `Counters.TOTAL_FREED_*` ids
/// (jit.py) to the matching field on a
/// [`CpuTotalTracker`].  Mirrors `jitprof.py:105-106`:
///
/// ```python
/// if num >= Counters.TOTAL_COMPILED_LOOPS:
///     return getattr(self.cpu.tracker, Counters.counter_names[num])
/// ```
///
/// Enforce the single-thread contract documented on [`TimingState`]:
/// once a thread has pushed onto the empty stack, only that thread
/// may push/pop until the stack drains back to empty.  Same-thread
/// re-entry (nested `start_event`/`end_event` on one thread) is
/// fine; a *different* thread arriving while `owner_thread.is_some()`
/// would interleave the LIFO stack and trip the `BROKEN PROFILER
/// DATA!` mismatch detector later.
///
/// Debug builds panic immediately with the offending caller name so
/// the cross-thread bug is visible at the actual misuse site.
/// Release builds skip the check (the caller is expected to
/// serialise profiler operations at a higher level, matching PyPy's
/// GIL-equivalent invariant).
fn check_or_claim_owner_thread(state: &mut TimingState, caller: &'static str) {
    if cfg!(debug_assertions) {
        let here = std::thread::current().id();
        match state.owner_thread {
            None => state.owner_thread = Some(here),
            Some(existing) if existing == here => {}
            Some(existing) => {
                panic!(
                    "JitProfiler {caller} from thread {here:?} while stack is owned by \
                     {existing:?}; profiler operations must be serialised at the caller \
                     (see TimingState single-thread contract)"
                );
            }
        }
    }
}

/// Caller must restrict `kind` to the four `TOTAL_*` ids
/// ([`is_cpu_tracker_kind`]); any other id panics on both debug and
/// release builds, matching the `assert!` strictness of
/// [`JitProfiler::count`] / [`JitProfiler::count_ops`] (PyPy raises
/// `IndexError` / `AttributeError` for the equivalent paths).
fn cpu_tracker_field(tracker: &CpuTotalTracker, kind: i32) -> &AtomicUsize {
    match kind {
        counters::TOTAL_COMPILED_LOOPS => &tracker.total_compiled_loops,
        counters::TOTAL_COMPILED_BRIDGES => &tracker.total_compiled_bridges,
        counters::TOTAL_FREED_LOOPS => &tracker.total_freed_loops,
        counters::TOTAL_FREED_BRIDGES => &tracker.total_freed_bridges,
        _ => panic!("cpu_tracker_field({kind}) — not a TOTAL_* id"),
    }
}

/// debug.py `debug_start("jit-tracing")` / `debug_start("jit-backend")`
/// channel name for a `Counters.*` event id.  Returns `None` for events
/// that don't have a paired debug scope upstream (count-only kinds like
/// OPS / ABORT_*).  Used by [`JitProfiler::start_event`] /
/// [`JitProfiler::end_event`] to emit grep-able scope markers under the
/// single `MAJIT_LOG` switch (see `memmgr.rs` TODO
/// note for the channel-registry deferral).
fn debug_channel_for_event(event: i32) -> Option<&'static str> {
    match event {
        counters::TRACING => Some("jit-tracing"),
        counters::BACKEND => Some("jit-backend"),
        _ => None,
    }
}

/// jitprof.py `Profiler._print_line_time` layout:
/// `"%s:%s\t%d\t%f" % (string, " " * max(0, 13-len(string)), i, tim)`.
fn fmt_line_time(string: &str, i: usize, tim: f64) -> String {
    let pad = " ".repeat(13usize.saturating_sub(string.len()));
    format!("{string}:{pad}\t{i}\t{tim:.6}")
}

/// jitprof.py `Profiler._print_intline` layout:
/// `string + ':' + " " * max(0, 16-len(string)) + '\t' + str(i)`.
fn fmt_intline(string: &str, i: usize) -> String {
    let pad = " ".repeat(16usize.saturating_sub(string.len()));
    format!("{string}:{pad}\t{i}")
}

/// Plain-old-data snapshot of [`JitProfiler`].
///
/// Used by debug printers, tests, and the `JitStats` view. Field order
/// mirrors RPython's `_print_stats` (jitprof.py) so the
/// eventual `print_stats` port can iterate on a fixed layout.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct JitProfilerSnapshot {
    pub tracing: usize,
    pub backend: usize,
    pub ops: usize,
    pub heapcached_ops: usize,
    pub recorded_ops: usize,
    pub guards: usize,
    pub opt_ops: usize,
    pub opt_guards: usize,
    pub opt_guards_shared: usize,
    pub opt_forcings: usize,
    pub opt_vectorize_try: usize,
    pub opt_vectorized: usize,
    pub abort_too_long: usize,
    pub abort_bridge: usize,
    pub abort_bad_loop: usize,
    pub abort_escape: usize,
    pub abort_force_quasiimmut: usize,
    pub abort_segmented_trace: usize,
    pub force_virtualizables: usize,
    pub nvirtuals: usize,
    pub nvholes: usize,
    pub nvreused: usize,
    pub calls: usize,
    pub tracing_time_ns: u64,
    pub backend_time_ns: u64,
}

/// jitprof.py `BrokenProfilerData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrokenProfilerData;

impl std::fmt::Display for BrokenProfilerData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BrokenProfilerData")
    }
}

impl std::error::Error for BrokenProfilerData {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_profiler_mirrors_noop_surface() {
        let prof = EmptyProfiler::default();
        assert!(prof.initialized);
        prof.start();
        prof.start_tracing();
        prof.count(counters::OPS, 5);
        prof.count_ops(OpCode::IntAdd, counters::RECORDED_OPS);
        prof.end_tracing();
        prof.end_backend();
        prof.finish();
        assert_eq!(prof.get_counter(counters::OPS), 0);
        assert_eq!(prof.get_times(counters::TRACING), 0.0);
    }

    #[test]
    fn profiler_alias_exposes_jit_profiler() {
        let prof = Profiler::default();
        prof.start();
        prof.count(counters::OPS, 1);
        assert_eq!(prof.get_counter(counters::OPS), Some(1));
    }

    #[test]
    fn broken_profiler_data_is_error_marker() {
        let err = BrokenProfilerData;
        assert_eq!(err.to_string(), "BrokenProfilerData");
    }

    #[test]
    fn count_ops_increments_kind_bucket_and_calls_only_on_call_recorded_ops() {
        // jitprof.py count_ops contract: counters[kind] += 1, and if the op
        // is a CALL_* AND kind == RECORDED_OPS, calls += 1.  Other
        // (kind, opnum) combinations leave `calls` untouched.
        let prof = JitProfiler::default();
        // OPS path on a non-call: ops += 1, calls unchanged.
        prof.count_ops(OpCode::IntAdd, counters::OPS);
        assert_eq!(prof.ops.load(Ordering::Relaxed), 1);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 0);
        // OPS path on a CALL_*: kind != RECORDED_OPS so calls untouched
        // (jitprof.py:121 only bumps calls on the RECORDED_OPS branch).
        prof.count_ops(OpCode::CallI, counters::OPS);
        assert_eq!(prof.ops.load(Ordering::Relaxed), 2);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 0);
        // RECORDED_OPS path on a non-call: recorded_ops += 1, calls untouched.
        prof.count_ops(OpCode::IntAdd, counters::RECORDED_OPS);
        assert_eq!(prof.recorded_ops.load(Ordering::Relaxed), 1);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 0);
        // RECORDED_OPS + CALL_*: both recorded_ops and calls bump.
        prof.count_ops(OpCode::CallI, counters::RECORDED_OPS);
        assert_eq!(prof.recorded_ops.load(Ordering::Relaxed), 2);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 1);
        // HEAPCACHED_OPS / GUARDS independent buckets.
        prof.count_ops(OpCode::PtrEq, counters::HEAPCACHED_OPS);
        prof.count_ops(OpCode::GuardTrue, counters::GUARDS);
        assert_eq!(prof.heapcached_ops.load(Ordering::Relaxed), 1);
        assert_eq!(prof.guards.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn count_routes_every_counters_id_to_its_atomic_field() {
        // Walk every Counters.* id `count` should accept and verify it
        // lands in the matching atomic.  Anchors the field_for_kind
        // dispatch table — a future Counters addition without a matching
        // arm is caught here.
        let prof = JitProfiler::default();
        for (kind, expected) in [
            (counters::TRACING, &prof.tracing),
            (counters::BACKEND, &prof.backend),
            (counters::OPS, &prof.ops),
            (counters::HEAPCACHED_OPS, &prof.heapcached_ops),
            (counters::RECORDED_OPS, &prof.recorded_ops),
            (counters::GUARDS, &prof.guards),
            (counters::OPT_OPS, &prof.opt_ops),
            (counters::OPT_GUARDS, &prof.opt_guards),
            (counters::OPT_GUARDS_SHARED, &prof.opt_guards_shared),
            (counters::OPT_FORCINGS, &prof.opt_forcings),
            (counters::OPT_VECTORIZE_TRY, &prof.opt_vectorize_try),
            (counters::OPT_VECTORIZED, &prof.opt_vectorized),
            (counters::ABORT_TOO_LONG, &prof.abort_too_long),
            (counters::ABORT_BRIDGE, &prof.abort_bridge),
            (counters::ABORT_BAD_LOOP, &prof.abort_bad_loop),
            (counters::ABORT_ESCAPE, &prof.abort_escape),
            (
                counters::ABORT_FORCE_QUASIIMMUT,
                &prof.abort_force_quasiimmut,
            ),
            (counters::ABORT_SEGMENTED_TRACE, &prof.abort_segmented_trace),
            (counters::FORCE_VIRTUALIZABLES, &prof.force_virtualizables),
            (counters::NVIRTUALS, &prof.nvirtuals),
            (counters::NVHOLES, &prof.nvholes),
            (counters::NVREUSED, &prof.nvreused),
        ] {
            let before = expected.load(Ordering::Relaxed);
            prof.count(kind, 3);
            assert_eq!(
                expected.load(Ordering::Relaxed),
                before + 3,
                "kind {kind} did not land in the expected atomic field",
            );
        }
    }

    #[test]
    fn start_end_on_one_thread_drains_owner_thread_for_the_next_caller() {
        // `TimingState.owner_thread` is set on the first push into an
        // empty stack and cleared when the stack drains back to empty.
        // Subsequent callers (from any thread) must be able to claim
        // ownership again — otherwise normal sequential use across
        // worker threads would trip the cross-thread check.
        let prof = JitProfiler::default();
        prof.start();
        prof.start_tracing();
        prof.end_tracing();
        // Stack is now empty; a fresh start from another thread (or
        // the same thread) must succeed.
        let prof_arc = Arc::new(prof);
        let prof_clone = Arc::clone(&prof_arc);
        std::thread::spawn(move || {
            prof_clone.start_backend();
            prof_clone.end_backend();
        })
        .join()
        .expect("worker should complete without panic");
        // Bookkeeping shows both events landed.
        let snap = prof_arc.snapshot();
        assert_eq!(snap.tracing, 1);
        assert_eq!(snap.backend, 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "JitProfiler")]
    fn cross_thread_start_event_while_stack_owned_panics_in_debug() {
        // The owner_thread guard exists to catch the cross-thread
        // interleaving bug the previous Mutex-only design hid: thread
        // A opens TRACING, thread B opens BACKEND, then A drops first
        // and pops B's BACKEND (mismatch → BROKEN PROFILER DATA).
        // Detect the bug at the second thread's push instead.
        let prof = Arc::new(JitProfiler::default());
        prof.start();
        prof.start_tracing();
        let prof_clone = Arc::clone(&prof);
        let join = std::thread::spawn(move || {
            prof_clone.start_backend();
        });
        // The worker thread's `start_backend` must panic on the owner
        // check.  Surface the panic to the test thread.
        let result = join.join();
        assert!(result.is_err(), "cross-thread push should panic in debug");
        // Drain on this thread so the panic doesn't leak owner_thread
        // (irrelevant for assertion shape but keeps the profiler
        // structurally tidy for any downstream code).
        prof.end_tracing();
        panic!("JitProfiler cross-thread check fired (expected by #[should_panic])");
    }

    #[test]
    fn count_with_unknown_kind_is_silent_noop() {
        // pyre's permissive variant of upstream's `counters[kind] += 1`:
        // an id outside the `Counters.*` set is silently ignored so a
        // miswired callsite cannot panic the profiler.
        let prof = JitProfiler::default();
        prof.count(-1, 5);
        prof.count(99, 5);
        prof.count_ops(OpCode::IntAdd, -7);
        let snap = prof.snapshot();
        assert_eq!(snap, JitProfilerSnapshot::default());
    }

    #[test]
    fn snapshot_reads_every_counter_independently() {
        // jitprof.py `_print_stats` reads each cnt[Counters.X]
        // one-by-one with no locking — `snapshot` must do the same.
        let prof = JitProfiler::default();
        prof.count(counters::OPS, 7);
        prof.count(counters::OPT_OPS, 11);
        prof.count(counters::NVIRTUALS, 13);
        prof.count_ops(OpCode::CallI, counters::RECORDED_OPS);
        let snap = prof.snapshot();
        assert_eq!(snap.ops, 7);
        assert_eq!(snap.opt_ops, 11);
        assert_eq!(snap.nvirtuals, 13);
        assert_eq!(snap.recorded_ops, 1);
        assert_eq!(snap.calls, 1);
    }

    #[test]
    fn get_counter_reads_via_kind_id() {
        // jitprof.py `get_counter(num)` — pyre returns Option to
        // signal unknown ids instead of upstream's IndexError.
        let prof = JitProfiler::default();
        prof.count(counters::ABORT_ESCAPE, 1);
        assert_eq!(prof.get_counter(counters::ABORT_ESCAPE), Some(1));
        assert_eq!(prof.get_counter(counters::OPS), Some(0));
        assert_eq!(prof.get_counter(-1), None);
    }

    #[test]
    fn start_end_timed_events_count_and_accumulate_elapsed_time() {
        // jitprof.py `_start`/`_end` contract: entering a nested
        // event charges elapsed time to the previously-active event; leaving
        // charges elapsed time to the ending event.
        let prof = JitProfiler::default();
        prof.start();
        prof.start_tracing();
        std::thread::sleep(Duration::from_millis(1));
        prof.start_backend();
        std::thread::sleep(Duration::from_millis(1));
        prof.end_backend();
        std::thread::sleep(Duration::from_millis(1));
        prof.end_tracing();

        let snap = prof.snapshot();
        assert_eq!(snap.tracing, 1);
        assert_eq!(snap.backend, 1);
        assert!(snap.tracing_time_ns > 0);
        assert!(snap.backend_time_ns > 0);
        assert!(prof.get_times(counters::TRACING).unwrap() > 0.0);
        assert_eq!(prof.get_times(counters::OPS), None);
    }

    #[test]
    #[should_panic(expected = "PyPy raises IndexError")]
    fn count_panics_on_total_compiled_loops_id() {
        // jitprof.py `self.counters[kind] += inc` raises
        // `IndexError` when `kind` is `TOTAL_COMPILED_LOOPS` (id 22)
        // because `self.counters` is sized `Counters.ncounters = 22`.
        // Pyre uses `assert!` (not `debug_assert!`) so the panic
        // fires in release builds too, matching upstream's crash on
        // a programmer-error caller.
        let prof = JitProfiler::default();
        prof.count(counters::TOTAL_COMPILED_LOOPS, 1);
    }

    #[test]
    #[should_panic(expected = "PyPy raises IndexError")]
    fn count_ops_panics_on_total_freed_bridges_id() {
        let prof = JitProfiler::default();
        prof.count_ops(OpCode::IntAdd, counters::TOTAL_FREED_BRIDGES);
    }

    #[test]
    fn set_cpu_tracker_routes_total_counters_to_bound_tracker() {
        // jitprof.py:105-106 contract: `Counters.TOTAL_*` reads go
        // through `self.cpu.tracker`.  After `set_cpu_tracker(arc)`,
        // writes to that Arc must be visible to the profiler's
        // `get_counter` and freed-bump helpers.  Two independent
        // profilers bound to separate trackers must NOT share state —
        // this is the regression under test.
        let prof_a = JitProfiler::default();
        let prof_b = JitProfiler::default();
        let tracker_a = Arc::new(CpuTotalTracker::default());
        let tracker_b = Arc::new(CpuTotalTracker::default());
        prof_a.set_cpu_tracker(Arc::clone(&tracker_a));
        prof_b.set_cpu_tracker(Arc::clone(&tracker_b));

        tracker_a.total_compiled_loops.store(3, Ordering::Relaxed);
        tracker_a.total_compiled_bridges.store(5, Ordering::Relaxed);
        prof_a.inc_freed_loop();
        prof_a.add_freed_bridges(7);

        assert_eq!(prof_a.get_counter(counters::TOTAL_COMPILED_LOOPS), Some(3));
        assert_eq!(
            prof_a.get_counter(counters::TOTAL_COMPILED_BRIDGES),
            Some(5)
        );
        assert_eq!(prof_a.get_counter(counters::TOTAL_FREED_LOOPS), Some(1));
        assert_eq!(prof_a.get_counter(counters::TOTAL_FREED_BRIDGES), Some(7));

        // `prof_b` saw none of those updates because it is bound to a
        // separate `CpuTotalTracker` instance.
        assert_eq!(prof_b.get_counter(counters::TOTAL_COMPILED_LOOPS), Some(0));
        assert_eq!(
            prof_b.get_counter(counters::TOTAL_COMPILED_BRIDGES),
            Some(0)
        );
        assert_eq!(prof_b.get_counter(counters::TOTAL_FREED_LOOPS), Some(0));
        assert_eq!(prof_b.get_counter(counters::TOTAL_FREED_BRIDGES), Some(0));
    }

    #[test]
    fn metainterp_static_data_default_initialises_profiler_with_zeroed_counters() {
        // pyjitpl.py:2199-2200 contract: every freshly built
        // `MetaInterpStaticData` carries a `Profiler()` with all
        // counters at 0 and exposes it as a public field so cross-crate
        // callers can hit it through the shared `Arc`.
        let sd = crate::pyjitpl::MetaInterpStaticData::new();
        let snap = sd.profiler.snapshot();
        assert_eq!(snap, JitProfilerSnapshot::default());
        // Update through the field and confirm the same struct
        // observes the bump (no shadowing into a separate counter).
        sd.profiler.count_ops(OpCode::IntAdd, counters::OPS);
        assert_eq!(sd.profiler.ops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn print_line_layouts_match_print_stats() {
        // `"%s:%s\t%d\t%f"` with `" " * max(0, 13-len(string))`.
        assert_eq!(
            fmt_line_time("Tracing", 2, 0.5),
            "Tracing:      \t2\t0.500000"
        );
        assert_eq!(fmt_intline("ops", 42), "ops:             \t42");
        // A label longer than 16 chars gets no padding (max(0, ..)).
        assert_eq!(
            fmt_intline("abort: trace too long", 1),
            "abort: trace too long:\t1"
        );
        // The calls line keeps its two-space indent inside the label.
        assert_eq!(fmt_intline("  calls", 7), "  calls:         \t7");
    }

    #[test]
    fn finish_marks_tk_and_prints_without_panicking() {
        let prof = JitProfiler::default();
        prof.start();
        prof.count_ops(OpCode::IntAdd, counters::OPS);
        // `finish()` = `self.tk = timer(); print_stats()`; with MAJIT_LOG
        // unset the print side is a no-op, the tk mark still lands.
        prof.finish();
        let state = prof.timing.lock();
        assert!(state.starttime.is_some());
        assert!(state.tk.is_some());
    }

    #[test]
    fn metainterp_staticdata_arc_lets_cross_crate_callers_share_one_profiler() {
        // The Arc<MetaInterpStaticData> on MetaInterp.staticdata is what
        // TraceCtx / heapcache / vector pass clone to reach the same
        // counter sink (RPython parity: `self.metainterp_sd.profiler`).
        // This test exercises the same shape: clone the Arc, bump
        // through the clone, observe via the original.
        let sd = std::sync::Arc::new(crate::pyjitpl::MetaInterpStaticData::new());
        let cross_crate_handle = sd.clone();
        cross_crate_handle
            .profiler
            .count_ops(OpCode::CallI, counters::RECORDED_OPS);
        cross_crate_handle.profiler.count(counters::NVIRTUALS, 4);
        let snap = sd.profiler.snapshot();
        assert_eq!(snap.recorded_ops, 1);
        assert_eq!(snap.calls, 1);
        assert_eq!(snap.nvirtuals, 4);
    }
}
