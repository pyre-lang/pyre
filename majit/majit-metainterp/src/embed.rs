//! Per-run JIT census for an embedder.
//!
//! An interpreter that drives [`JitDriver`] needs to know what its JIT actually
//! did: a green test suite, agreement with the untraced tier and an exact
//! answer are all satisfied by an interpreter that never compiled anything.
//! The evidence lives behind six driver callbacks and a set of process-global
//! diagnostic slots. [`Census`] wires those up once so an embedder does not
//! rebuild the counters, the callback closures and the serializing lock per
//! module.
//!
//! # The window is the point
//!
//! The counters behind this module are process-global and cumulative, because
//! the callbacks are `'static` closures that cannot borrow a caller's local
//! state. An absolute read therefore answers "since this process started",
//! which is not the question a run asks. [`Census::begin`] opens a window and
//! [`Census::counts`] reports the DELTA across it.
//!
//! Two rules make that delta trustworthy, and both were learned from getting
//! them wrong:
//!
//! * The serializing lock is a process-global [`Mutex`], not a `thread_local!`.
//!   Its job is to keep two concurrently running consumers — parallel test
//!   threads, most often — out of each other's window. A thread-local lock is
//!   uncontended by construction and silently does nothing at all.
//! * The cumulative counters are diffed, never zeroed, so a window cannot
//!   destroy an enclosing reader's totals. The "last compiled body" fields
//!   cannot be diffed, so [`Census::begin`] resets those instead — and it holds
//!   the window lock across the reset AND the read, which is what makes that
//!   reset safe. A reset outside a held lock hands each of two overlapping
//!   consumers a fraction of the truth.

use parking_lot::{Mutex, MutexGuard};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{JitDriver, JitState, LoopBodyShape};

/// Loops compiled, from the compile-loop callback.
static COMPILES: AtomicUsize = AtomicUsize::new(0);
/// Guards that failed back into the runtime, from the guard-failure callback.
static GUARD_FAILURES: AtomicUsize = AtomicUsize::new(0);
/// Traces abandoned before they closed, from the trace-abort callback.
static TRACE_ABORTS: AtomicUsize = AtomicUsize::new(0);
/// Calls that entered compiled code, from the compiled-entry callback.
static COMPILED_ENTRIES: AtomicUsize = AtomicUsize::new(0);
/// Recorded and optimized op counts, summed over every compiled loop.
static TRACE_OPS_BEFORE: AtomicUsize = AtomicUsize::new(0);
static TRACE_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);
/// Bridges compiled, from the compile-bridge callback.
///
/// A callback rather than a read of `JitDriver::get_stats().bridges_compiled`,
/// which is the same number by a route that cannot be windowed: the tally lives
/// on the driver, so a reader that only reaches drivers it has already retired
/// reports zero for a pool whose drivers are all still alive. The hook fires on
/// the line after each of the two `stats.bridges_compiled += 1` sites, inside
/// the same block and under no extra condition, so the two counts cannot come
/// apart.
static BRIDGES: AtomicUsize = AtomicUsize::new(0);
/// Compilation panics swallowed by a driver, from the compile-panic callback.
///
/// A hook of its own rather than `on_compile_error`, which the same site also
/// fires: that one fires for every RECOVERABLE compile failure, so counting it
/// would not be this number.
static PANICS: AtomicUsize = AtomicUsize::new(0);

/// The last compiled body's optimized op count and shape.
///
/// Not cumulative, so [`Census::begin`] resets them rather than diffing them.
/// The shape is held as its two flags rather than the struct so the recording
/// stays lock-free on the compile path.
static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);
static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

/// Serializes measurement windows against each other.
///
/// Process-global on purpose: the counters it protects are, and the consumer
/// this exists for is a parallel test runner entering the same JIT from several
/// threads at once.
static WINDOW_LOCK: Mutex<()> = Mutex::new(());

/// The opcode list of every loop compiled since the last [`Census::begin`] or
/// [`Census::reset`], in compile order. An RCA instrument: the counters above
/// say HOW MANY ops a trace kept, this says WHICH.
static COMPILED_OPCODE_LOG: Mutex<Vec<Vec<majit_ir::OpCode>>> = Mutex::new(Vec::new());

/// One reading of what the JIT did.
///
/// Every field except the two `last_*` ones is a count over the window it was
/// read in. `bridges_compiled` and `internal_compile_panics` are the exception
/// in a second way as well: they have no callback and only appear here for a
/// driver that was handed to [`Census::absorb`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CensusCounts {
    pub loops_compiled: usize,
    pub bridges_compiled: usize,
    pub loops_aborted: usize,
    pub guard_failures: usize,
    /// Non-zero means a trace was dropped by a panic inside compilation and the
    /// tier silently fell back to the untraced path for it. Nothing else
    /// reports this: a run that stops compiling still answers correctly, so
    /// every other counter here stays plausible.
    pub internal_compile_panics: usize,
    pub trace_ops_before: usize,
    pub trace_ops_after: usize,
    /// The only field that separates "an artifact exists" from "an artifact
    /// ran".
    pub compiled_entries: usize,
    /// Optimized op count of the LAST compiled body in the window.
    ///
    /// `loops_compiled > 0` is necessary but not sufficient evidence of a
    /// working tier: an entirely empty dispatch still compiles a trace, one
    /// whose whole optimized body is `Finish()`. A compile counter counts
    /// TRACES, not WORK.
    pub last_ops_after: usize,
    /// Shape of the LAST compiled body in the window. Its [`Default`] value is
    /// also what "nothing compiled in this window" reads as — see
    /// [`LoopBodyShape::why_not`], whose no-`Jump` arm is phrased for
    /// both cases.
    pub last_loop_body_shape: LoopBodyShape,
}

impl fmt::Display for CensusCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "loops_compiled={} bridges_compiled={} loops_aborted={} \
             guard_failures={} internal_compile_panics={} trace_ops_before={} \
             trace_ops_after={} compiled_entries={} last_ops_after={} \
             last_closes_a_loop={}",
            self.loops_compiled,
            self.bridges_compiled,
            self.loops_aborted,
            self.guard_failures,
            self.internal_compile_panics,
            self.trace_ops_before,
            self.trace_ops_after,
            self.compiled_entries,
            self.last_ops_after,
            self.last_loop_body_shape.closes_a_loop(),
        )
    }
}

/// An open measurement window over the process-global JIT counters.
///
/// Hold one for as long as the run being measured, then read it:
///
/// ```ignore
/// let census = Census::begin();
/// let answer = run_the_interpreter();
/// assert!(census.counts().loops_compiled > 0, "{census}");
/// ```
///
/// ⚠ NOT REENTRANT. [`Census::begin`] takes a plain [`Mutex`], so opening a
/// second window on a thread that already holds one deadlocks. Where a test
/// helper opens the window, every test that can compile has to go through that
/// helper and none of them may call another.
///
/// ⚠ Bind it to a NAMED local, `_census` included. `let _ = Census::begin();`
/// drops the window on the spot and leaves the run unserialized, which fails
/// only under contention and only sometimes.
pub struct Census {
    /// Cumulative counters at [`Census::begin`], subtracted at read time.
    base: RawCounts,
    /// Abort-reason tallies at [`Census::begin`], for the same reason.
    aborts_before: Vec<(&'static str, u64)>,
    /// Held for the window's whole life. Dropping the [`Census`] closes it.
    _window: MutexGuard<'static, ()>,
}

/// The cumulative half of a reading, snapshotted for the later subtraction.
#[derive(Clone, Copy, Debug, Default)]
struct RawCounts {
    compiles: usize,
    bridges: usize,
    aborts: usize,
    guard_failures: usize,
    panics: usize,
    ops_before: usize,
    ops_after: usize,
    compiled_entries: usize,
}

impl RawCounts {
    fn read() -> Self {
        Self {
            compiles: COMPILES.load(Ordering::Relaxed),
            bridges: BRIDGES.load(Ordering::Relaxed),
            aborts: TRACE_ABORTS.load(Ordering::Relaxed),
            guard_failures: GUARD_FAILURES.load(Ordering::Relaxed),
            panics: PANICS.load(Ordering::Relaxed),
            ops_before: TRACE_OPS_BEFORE.load(Ordering::Relaxed),
            ops_after: TRACE_OPS_AFTER.load(Ordering::Relaxed),
            compiled_entries: COMPILED_ENTRIES.load(Ordering::Relaxed),
        }
    }
}

impl Census {
    /// Wire every driver callback the census reads into the process-global
    /// counters.
    ///
    /// The six hooks are `set_on_compile_loop`, `set_on_compile_bridge`,
    /// `set_on_guard_failure`, `set_on_trace_abort`, `set_on_compiled_entry`
    /// and `set_on_internal_compile_panic`. Each holds ONE
    /// closure, so this replaces whatever was installed before it. An embedder
    /// that needs the compiled body's shape reads
    /// [`CensusCounts::last_loop_body_shape`] rather than installing its own
    /// `set_on_compile_loop`, which this would then displace.
    ///
    /// Install it on every driver whose activity should be counted — the
    /// counters are shared, the callbacks are not.
    pub fn install<S: JitState>(driver: &mut JitDriver<S>) {
        driver.set_on_compile_loop(|_green_key, ops_before, ops_after, opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
            TRACE_OPS_BEFORE.fetch_add(ops_before, Ordering::Relaxed);
            TRACE_OPS_AFTER.fetch_add(ops_after, Ordering::Relaxed);
            LAST_OPS_AFTER.store(ops_after, Ordering::Relaxed);
            COMPILED_OPCODE_LOG.lock().push(opcodes.to_vec());
            let shape = LoopBodyShape::of(opcodes);
            LAST_HAS_JUMP.store(shape.has_jump, Ordering::Relaxed);
            LAST_ALWAYS_FAILS.store(shape.has_always_fails, Ordering::Relaxed);
        });
        driver.set_on_compile_bridge(|_green_key, _descr_id, _num_ops| {
            BRIDGES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_guard_failure(|_green_key, _descr_id, _count| {
            GUARD_FAILURES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_trace_abort(|_green_key, _permanent| {
            TRACE_ABORTS.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_compiled_entry(|_green_key, _target_pc| {
            COMPILED_ENTRIES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_internal_compile_panic(|_green_key| {
            PANICS.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// Open a measurement window: serialize against any other window, snapshot
    /// the cumulative counters, and reset the non-cumulative ones.
    ///
    /// Blocks until any window open on another thread closes. See the type's
    /// reentrancy and binding warnings.
    #[must_use = "the window closes when the returned Census is dropped"]
    pub fn begin() -> Self {
        let window = WINDOW_LOCK.lock();
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        COMPILED_OPCODE_LOG.lock().clear();
        Self {
            base: RawCounts::read(),
            aborts_before: abort_reasons(),
            _window: window,
        }
    }

    /// What the JIT did since [`Census::begin`].
    pub fn counts(&self) -> CensusCounts {
        let now = RawCounts::read();
        CensusCounts {
            loops_compiled: now.compiles.saturating_sub(self.base.compiles),
            bridges_compiled: now.bridges.saturating_sub(self.base.bridges),
            loops_aborted: now.aborts.saturating_sub(self.base.aborts),
            guard_failures: now.guard_failures.saturating_sub(self.base.guard_failures),
            internal_compile_panics: now.panics.saturating_sub(self.base.panics),
            trace_ops_before: now.ops_before.saturating_sub(self.base.ops_before),
            trace_ops_after: now.ops_after.saturating_sub(self.base.ops_after),
            compiled_entries: now
                .compiled_entries
                .saturating_sub(self.base.compiled_entries),
            last_ops_after: LAST_OPS_AFTER.load(Ordering::Relaxed),
            last_loop_body_shape: LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        }
    }

    /// The abort reasons that fired since [`Census::begin`], as `label=delta`.
    ///
    /// Empty when nothing aborted, so a caller can print it unconditionally in
    /// a failure message and a quiet window stays quiet. See [`abort_reasons`]
    /// for what the labels do and do not distinguish.
    pub fn abort_reasons_since(&self) -> String {
        render_abort_delta(&self.aborts_before, &abort_reasons())
    }

    /// Absolute counters since process start, or since the last
    /// [`Census::reset`]. Takes no lock, so it is safe to call from inside an
    /// open window — and is a total, not a window, which is why the assertions
    /// a run makes belong on [`Census::counts`] instead.
    pub fn totals() -> CensusCounts {
        let now = RawCounts::read();
        CensusCounts {
            loops_compiled: now.compiles,
            bridges_compiled: now.bridges,
            loops_aborted: now.aborts,
            guard_failures: now.guard_failures,
            internal_compile_panics: now.panics,
            trace_ops_before: now.ops_before,
            trace_ops_after: now.ops_after,
            compiled_entries: now.compiled_entries,
            last_ops_after: LAST_OPS_AFTER.load(Ordering::Relaxed),
            last_loop_body_shape: LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        }
    }

    /// Zero every counter this module owns.
    ///
    /// For a caller that reads totals rather than windows. It moves the ground
    /// under any window that is already open, so a windowed reader should have
    /// no reason to call it — [`Census::begin`] already gives that reader a
    /// fresh zero without disturbing anyone else.
    pub fn reset() {
        for counter in [
            &COMPILES,
            &BRIDGES,
            &GUARD_FAILURES,
            &TRACE_ABORTS,
            &COMPILED_ENTRIES,
            &TRACE_OPS_BEFORE,
            &TRACE_OPS_AFTER,
            &PANICS,
            &LAST_OPS_AFTER,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        COMPILED_OPCODE_LOG.lock().clear();
    }

    /// A copy of the opcode log -- see [`COMPILED_OPCODE_LOG`].
    pub fn compiled_opcode_log() -> Vec<Vec<majit_ir::OpCode>> {
        COMPILED_OPCODE_LOG.lock().clone()
    }
}

impl fmt::Display for Census {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.counts())
    }
}

/// The degraded dispatch arms recorded for `interp`, sorted by arm name.
///
/// The process-wide registry holds every machine's arms, so a reader that wants
/// one machine's has to filter; sorting makes the result comparable against a
/// written-down set whatever order the portal happened to install them in.
///
/// ⚠ This is NOT a windowed reading and must not be made one, which is what
/// separates it from every counter in this module. The registry records a
/// per-ARM fact at portal-install time and deduplicates by content, so it
/// describes what the lowerer can express about a machine, not what one run
/// did. Diffing it across a window would report "installed during this window",
/// a different and far less useful property.
pub fn degraded_dispatch_arm_names(interp: &str) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = crate::degraded_dispatch_arms()
        .into_iter()
        .filter(|arm| arm.interp == interp)
        .map(|arm| arm.arm)
        .collect();
    names.sort_unstable();
    names
}

/// The degraded arms recorded for `interp` paired with their refusal family,
/// sorted by arm name.
///
/// Strictly more than [`degraded_dispatch_arm_names`]: an arm can keep
/// degrading under a different mechanism, which moves the family while the name
/// set sits still. Only the FIRST refusal in each arm's reason is classified —
/// that is the outermost blocker, the one that stopped lowering.
pub fn degraded_dispatch_arm_causes(interp: &str) -> Vec<(&'static str, crate::RefusalKind)> {
    let mut causes: Vec<(&'static str, crate::RefusalKind)> = crate::degraded_dispatch_arms()
        .into_iter()
        .filter(|arm| arm.interp == interp)
        .map(|arm| (arm.arm, crate::refusal_kind(arm.reason)))
        .collect();
    causes.sort_unstable_by_key(|(arm, _)| *arm);
    causes
}

/// The refusal reason recorded for one of `interp`'s arms, if that arm
/// degraded.
///
/// Looked up by arm name rather than by position: the registry's order is the
/// portal's install order, so an index is a pin on something nobody chose.
pub fn degraded_dispatch_arm_reason(interp: &str, arm: &str) -> Option<&'static str> {
    crate::degraded_dispatch_arms()
        .into_iter()
        .find(|entry| entry.interp == interp && entry.arm == arm)
        .map(|entry| entry.reason)
}

/// Panic unless `interp`'s `arm` degraded AND its reason contains `needle`.
///
/// One level below [`assert_degraded_dispatch_arm_causes`]: the family says
/// which MECHANISM refused the arm, and this says which SOURCE it refused on.
/// A refusal can keep its family while moving to a different statement, which
/// means the gap being tracked is not the one the pin was written for.
///
/// A substring, never the whole reason — the refusal is rendered with the
/// lowerer's own spacing, which is not a thing a gate should be pinned to.
pub fn assert_degraded_dispatch_arm_reason_contains(interp: &str, arm: &str, needle: &str) {
    match degraded_dispatch_arm_reason(interp, arm) {
        Some(reason) => assert!(
            reason.contains(needle),
            "`{interp}`'s `{arm}` no longer refuses on `{needle}`, so the pin \
             tracks a gap that has moved: {reason}"
        ),
        None => panic!(
            "`{interp}` has no degraded arm named `{arm}`, so there is no \
             refusal to read `{needle}` out of. Degraded arms: {:?}",
            degraded_dispatch_arm_names(interp)
        ),
    }
}

/// Panic unless `interp`'s portal was installed AND its degraded arms are
/// EXACTLY `expected`.
///
/// **An equality over a named set, never an emptiness check.** A machine with a
/// known lowering gap has a non-empty degraded set today, so demanding
/// emptiness there fails on day one and the only available response is to
/// weaken the gate into uselessness. Pinning the set keeps both directions
/// live: a NEW name means an arm silently stopped lowering and every trace
/// reaching it aborts, and a MISSING name means that arm lowers again, so the
/// gap the pin was tracking is closed and the surrounding gate can be
/// strengthened. An emptiness check reports only the first of those, and only
/// where the answer is already zero.
///
/// Pass `&[]` for a machine that should have no degraded arm at all;
/// [`crate::assert_no_degraded_dispatch_arms`] is that spelling.
///
/// The portal check is what makes an empty `expected` mean anything. An empty
/// registry is also what a machine whose portal was never built produces — the
/// same shape as the defect the gate exists to report, one level up — so the
/// arm census supplies the denominator and the two failures get two different
/// messages instead of one silent pass.
///
/// Call it after whatever installs the portal. The facts are recorded at
/// install, not at trace time, so running the machine is not required — but
/// nothing is recorded until the portal is built at least once.
pub fn assert_degraded_dispatch_arms(interp: &str, expected: &[&str]) {
    let arms = assert_portal_installed(interp, expected.is_empty());
    let degraded = degraded_dispatch_arm_names(interp);
    assert!(
        degraded == expected,
        "`{interp}`'s arms that lowered to an abort stub are {degraded:?}, not \
         the pinned {expected:?} — out of the {arms} arm(s) its portal emitted. \
         Every trace that reaches a stub aborts, once per threshold, forever. A \
         NEW name means an arm silently stopped lowering; a MISSING name means \
         that arm lowers again, so the pin is stale and whatever it was blocking \
         may now be possible."
    );
}

/// Panic unless `interp`'s portal was installed AND its degraded arms and their
/// refusal families are EXACTLY `expected`.
///
/// The companion to [`assert_degraded_dispatch_arms`] and strictly stronger,
/// since the pairs pin the names too. Both are worth asserting where a machine
/// has a known gap: they fail for different reasons and the two messages say so
/// — one reports that the SET moved, this one that a set which did not move is
/// now being refused by a different mechanism.
pub fn assert_degraded_dispatch_arm_causes(interp: &str, expected: &[(&str, crate::RefusalKind)]) {
    assert_portal_installed(interp, expected.is_empty());
    let causes = degraded_dispatch_arm_causes(interp);
    let matches = causes.len() == expected.len()
        && causes
            .iter()
            .zip(expected)
            .all(|((arm, kind), (want_arm, want_kind))| arm == want_arm && kind == want_kind);
    assert!(
        matches,
        "`{interp}`'s degraded dispatch arms are refused by {causes:?}, not the \
         pinned {expected:?}. A family that moved while its arm name did not \
         means a different mechanism is refusing that arm now; \
         `RefusalKind::Unclassified` means a refusal family exists that nothing \
         classifies yet."
    );
}

/// The arm count `interp`'s portal recorded, or a panic naming the machines
/// that did record one.
///
/// `expecting_none` only shapes the message: a pin that expects degraded arms
/// would fail anyway on the empty registry, but it would fail saying "the set
/// moved" when the truth is that nothing was ever measured.
fn assert_portal_installed(interp: &str, expecting_none: bool) -> usize {
    let census = crate::dispatch_arm_census();
    match census.iter().find(|entry| entry.interp == interp) {
        Some(entry) => entry.arms,
        None => {
            let consequence = if expecting_none {
                "so an empty degraded list says nothing about it"
            } else {
                "so the degraded list below is empty because nothing was measured, \
                 not because the pinned arms started lowering"
            };
            panic!(
                "no dispatch-arm census for `{interp}`: its portal was never \
                 installed in this process, {consequence}. Build the dispatch \
                 JitCode (or run the machine) first. Recorded machines: {:?}",
                census.iter().map(|e| e.interp).collect::<Vec<_>>()
            );
        }
    }
}

/// Snapshot the abort-reason tallies as `(label, count)` pairs.
///
/// [`CensusCounts::loops_aborted`] counts aborts without saying why, and the
/// reason is not carried on the trace-abort callback — it survives only in the
/// `MC_DIAG` slots. Without this a gate can assert `loops_aborted == 0`, watch
/// it fail, and have no way to learn which reason fired.
///
/// Selection is by the `abrt_` label prefix rather than by the slot range those
/// labels currently occupy: a hard-coded range names the wrong counters the
/// moment a slot is added, and it does so silently.
///
/// ⚠ `abrt_bridge` is the UNCLASSIFIED bucket wearing a specific-sounding name.
/// Every abort whose reason was never staged falls back to the generic reason,
/// whose id is that slot, so a count there is not evidence of bridge activity.
/// [`Census::abort_reasons_since`] relabels it in its output for that reason.
///
/// ⚠ The slots are process-global, cumulative, and have no reset, so diff two
/// snapshots rather than reading one.
pub fn abort_reasons() -> Vec<(&'static str, u64)> {
    crate::MC_DIAG_LABELS
        .iter()
        .enumerate()
        .filter(|(_, label)| label.starts_with("abrt_"))
        .map(|(slot, label)| (*label, crate::mc_diag(slot)))
        .collect()
}

/// Render `after - before` over two [`abort_reasons`] snapshots, dropping the
/// slots that did not move.
pub fn render_abort_delta(before: &[(&'static str, u64)], after: &[(&'static str, u64)]) -> String {
    after
        .iter()
        .zip(before)
        .filter_map(|((label, now), (_, then))| {
            let delta = now.saturating_sub(*then);
            if delta == 0 {
                return None;
            }
            let name = if *label == "abrt_bridge" {
                "unclassified(abrt_bridge)"
            } else {
                label
            };
            Some(format!("{name}={delta}"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window read is a DELTA, so activity that happened before it opened
    /// belongs to nobody's window.
    #[test]
    fn a_window_reports_only_what_happened_inside_it() {
        let census = Census::begin();
        assert_eq!(census.counts(), CensusCounts::default());

        COMPILES.fetch_add(2, Ordering::Relaxed);
        TRACE_OPS_AFTER.fetch_add(11, Ordering::Relaxed);
        assert_eq!(census.counts().loops_compiled, 2);
        assert_eq!(census.counts().trace_ops_after, 11);

        drop(census);
        // A second window opened after the first closed starts from zero
        // without either of them having zeroed a shared counter.
        let next = Census::begin();
        assert_eq!(next.counts().loops_compiled, 0);
        assert!(Census::totals().loops_compiled >= 2);
    }

    /// The `last_*` fields cannot be diffed, so the window resets them — and a
    /// window with no compile in it must not inherit the previous one's body.
    #[test]
    fn the_last_body_fields_do_not_leak_across_windows() {
        {
            let _census = Census::begin();
            LAST_OPS_AFTER.store(42, Ordering::Relaxed);
            LAST_HAS_JUMP.store(true, Ordering::Relaxed);
        }
        let census = Census::begin();
        let counts = census.counts();
        assert_eq!(counts.last_ops_after, 0);
        assert!(!counts.last_loop_body_shape.closes_a_loop());
        assert!(counts.last_loop_body_shape.why_not().is_some());
    }

    #[test]
    fn abort_reason_labels_are_selected_by_prefix_not_by_slot_range() {
        let reasons = abort_reasons();
        assert!(!reasons.is_empty());
        assert!(reasons.iter().all(|(label, _)| label.starts_with("abrt_")));
    }

    #[test]
    fn a_quiet_abort_window_renders_empty_and_a_busy_one_names_its_slot() {
        let before = vec![("abrt_too_long", 3u64), ("abrt_bridge", 1u64)];
        assert_eq!(render_abort_delta(&before, &before), "");

        let after = vec![("abrt_too_long", 5u64), ("abrt_bridge", 4u64)];
        assert_eq!(
            render_abort_delta(&before, &after),
            "abrt_too_long=2 unclassified(abrt_bridge)=3"
        );
    }

    /// Register a synthetic machine so the registry assertions have a subject.
    ///
    /// The registry is process-wide, deduplicated by content and never cleared,
    /// so each test uses a name of its own rather than trying to reset it.
    fn record_machine(
        interp: &'static str,
        arms: usize,
        degraded: &[(&'static str, &'static str)],
    ) {
        crate::record_dispatch_arm_census(interp, arms);
        for (arm, reason) in degraded {
            crate::record_degraded_dispatch_arm(interp, arm, reason);
        }
    }

    /// The finding this gate exists for, as an executable claim: a machine with
    /// a known lowering gap can pin its set, and the emptiness-only spelling can
    /// only fail on it.
    #[test]
    fn a_known_gap_is_expressible_as_a_pinned_set() {
        record_machine(
            "PinnedGapState",
            4,
            &[("PUSHARG", "arm body cannot express this statement")],
        );
        assert_degraded_dispatch_arms("PinnedGapState", &["PUSHARG"]);
        assert_degraded_dispatch_arm_causes(
            "PinnedGapState",
            &[("PUSHARG", crate::RefusalKind::UnlowerableStmt)],
        );
    }

    #[test]
    #[should_panic(expected = "not the pinned []")]
    fn the_zero_spelling_cannot_express_a_known_gap() {
        record_machine(
            "ZeroSpellingState",
            4,
            &[("PUSHARG", "arm body cannot express this statement")],
        );
        crate::assert_no_degraded_dispatch_arms("ZeroSpellingState");
    }

    /// Install order is not the pin's business, so the reader sorts.
    #[test]
    fn arm_names_are_sorted_whatever_order_the_portal_installed_them() {
        record_machine(
            "SortedState",
            3,
            &[
                ("ROLL", "arm body cannot express this statement"),
                ("ALLOCATE", "arm body cannot express this statement"),
            ],
        );
        assert_eq!(
            degraded_dispatch_arm_names("SortedState"),
            ["ALLOCATE", "ROLL"]
        );
        assert_degraded_dispatch_arms("SortedState", &["ALLOCATE", "ROLL"]);
    }

    /// A machine whose portal was never built has an empty degraded list, which
    /// is the same shape as a healthy one. The denominator is what tells them
    /// apart, and it must do so for a NON-empty pin too — there the empty list
    /// would otherwise read as "the pinned arms started lowering".
    #[test]
    #[should_panic(expected = "nothing was measured")]
    fn an_uninstalled_portal_does_not_read_as_a_closed_gap() {
        assert_degraded_dispatch_arms("NeverInstalledState", &["PUSHARG"]);
    }

    #[test]
    #[should_panic(expected = "says nothing about it")]
    fn an_uninstalled_portal_does_not_read_as_a_clean_one() {
        crate::assert_no_degraded_dispatch_arms("NeverInstalledEmptyState");
    }

    /// The family says which mechanism refused; the reason says which source.
    #[test]
    fn a_refusal_is_readable_by_arm_name_not_by_install_position() {
        record_machine(
            "ReasonState",
            2,
            &[
                ("ROLL", "arm body cannot express this statement: pc += 1"),
                (
                    "ALLOCATE",
                    "arm body cannot express this statement: regs[n]",
                ),
            ],
        );
        // Looked up by name, so the portal's install order is not a pin.
        assert_degraded_dispatch_arm_reason_contains("ReasonState", "ALLOCATE", "regs[n]");
        assert_degraded_dispatch_arm_reason_contains("ReasonState", "ROLL", "pc += 1");
        assert!(degraded_dispatch_arm_reason("ReasonState", "NOSUCH").is_none());
    }

    #[test]
    #[should_panic(expected = "no degraded arm named")]
    fn a_reason_pin_on_an_arm_that_lowers_reports_that_rather_than_a_mismatch() {
        record_machine("LoweredArmState", 2, &[]);
        assert_degraded_dispatch_arm_reason_contains("LoweredArmState", "ROLL", "pc += 1");
    }

    /// A family can move while the name set sits still, which is the whole
    /// reason the causes gate is separate from the names one.
    #[test]
    #[should_panic(expected = "refused by")]
    fn a_cause_that_moved_under_an_unchanged_name_is_a_failure() {
        record_machine(
            "MovedCauseState",
            2,
            &[("ALLOCATE", "arm body cannot express this statement")],
        );
        assert_degraded_dispatch_arms("MovedCauseState", &["ALLOCATE"]);
        assert_degraded_dispatch_arm_causes(
            "MovedCauseState",
            &[("ALLOCATE", crate::RefusalKind::GreenWriteback)],
        );
    }

    #[test]
    fn display_names_every_field_it_reports() {
        let text = CensusCounts::default().to_string();
        for key in [
            "loops_compiled=",
            "bridges_compiled=",
            "loops_aborted=",
            "guard_failures=",
            "internal_compile_panics=",
            "trace_ops_before=",
            "trace_ops_after=",
            "compiled_entries=",
            "last_ops_after=",
            "last_closes_a_loop=",
        ] {
            assert!(text.contains(key), "{key} missing from {text}");
        }
    }
}
